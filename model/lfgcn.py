import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


from utils import constant, torch_utils
from entmax import entmax_bisect


class GCNClassifier(nn.Module):
    def __init__(self, opt, emb_matrix=None):
        print("running multilsr model")
        super(GCNClassifier, self).__init__()
        self.gcn_model = GCNRelationModel(opt, emb_matrix=emb_matrix)
        in_dim = opt['hidden_dim']
        self.classifer = nn.Linear(in_dim, opt['num_class'])
        self.opt = opt

    def forward(self, inputs):
        outputs, pooling_output = self.gcn_model(inputs)
        logits = self.classifer(outputs)
        return logits, pooling_output


class GCNRelationModel(nn.Module):
    def __init__(self, opt, emb_matrix=None):
        super(GCNRelationModel, self).__init__()
        self.opt = opt
        self.emb_matrix = emb_matrix

        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
        self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), opt['pos_dim']) if opt['pos_dim'] > 0 else None
        embeddings = (self.emb, self.pos_emb)
        self.init_embeddings()

        self.gcn = LSR(opt, embeddings)

        in_dim = opt['hidden_dim'] * 3
        layers = [nn.Linear(in_dim, opt['hidden_dim']), nn.ReLU()]

        for _ in range(self.opt['mlp_layers'] - 1):
            layers += [nn.Linear(opt['hidden_dim'], opt['hidden_dim']), nn.ReLU()]

        self.out_mlp = nn.Sequential(*layers)

    def init_embeddings(self):
        if self.emb_matrix is None:
            self.emb.weight.data[1:, :].uniform_(-1.0, 1.0)
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)
        # decide finetuning
        if self.opt['topn'] <= 0:
            print("Do not finetune word embedding layer.")
            self.emb.weight.requires_grad = False
        elif self.opt['topn'] < self.opt['vocab_size']:
            print("Finetune top {} word embeddings.".format(self.opt['topn']))
            self.emb.weight.register_hook(lambda x: torch_utils.keep_partial_grad(x, self.opt['topn']))
        else:
            print("Finetune all embeddings.")

    def forward(self, inputs):
        words, masks, pos, deprel, head, subj_pos, obj_pos = inputs

        h, pool_mask = self.gcn(inputs)

        subj_mask, obj_mask = subj_pos.eq(0).eq(0).unsqueeze(2), obj_pos.eq(0).eq(0).unsqueeze(2)  # invert mask
        pool_type = self.opt['pooling']
        masks = masks.unsqueeze(-1)
        pool_mask = torch.add(masks, pool_mask)

        h_out = pool(h, pool_mask, type=pool_type)
        subj_out = pool(h, subj_mask, type="max")
        obj_out = pool(h, obj_mask, type="max")
        outputs = torch.cat([h_out, subj_out, obj_out], dim=1)
        outputs = self.out_mlp(outputs)

        return outputs, h_out


class LSR(nn.Module):
    def __init__(self, opt, embeddings):
        super(LSR, self).__init__()
        self.opt = opt
        self.in_dim = opt['emb_dim'] + opt['pos_dim']
        self.emb, self.pos_emb = embeddings
        self.use_cuda = opt['cuda']
        self.mem_dim = opt['hidden_dim']

        self.use_sparsemax = False

        # rnn layer
        if self.opt.get('rnn', False):
            self.input_W_R = nn.Linear(self.in_dim, opt['rnn_hidden'])
            self.rnn = nn.LSTM(opt['rnn_hidden'], opt['rnn_hidden'], opt['rnn_layers'], batch_first=True, \
                               dropout=opt['rnn_dropout'], bidirectional=True)
            self.in_dim = opt['rnn_hidden'] * 2
            self.rnn_drop = nn.Dropout(opt['rnn_dropout'])  # use on last layer output
        self.input_W_G = nn.Linear(self.in_dim, self.mem_dim)

        self.in_drop = nn.Dropout(opt['input_dropout'])
        self.num_layers = opt['num_layers']

        self.layers = nn.ModuleList()

        self.alpha_list = list()

        self.heads = opt['heads']
        self.sublayer_first = opt['sublayer_first']
        self.sublayer_second = opt['sublayer_second']
        self.trees_num = opt['heads']

        # gcn layer
        for i in range(self.num_layers):
            for j in range(self.heads):
                self.alpha_list.append(torch.tensor(1.33, requires_grad=True).cuda())
            self.layers.append(StructuredAttention(self.mem_dim, self.trees_num))
            self.layers.append(MultiGraphConvLayer(opt, self.mem_dim, self.sublayer_first, self.trees_num))
            self.layers.append(MultiGraphConvLayer(opt, self.mem_dim, self.sublayer_second, self.trees_num))

        # TODO: 1 layer or 2 layer
        self.aggregate_W = nn.Linear(2 * self.num_layers * self.mem_dim, self.mem_dim)

    def encode_with_rnn(self, rnn_inputs, masks, batch_size):
        seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())
        h0, c0 = rnn_zero_state(batch_size, self.opt['rnn_hidden'], self.opt['rnn_layers'])
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def forward(self, inputs):
        words, masks, pos, deprel, head, subj_pos, obj_pos = inputs  # unpack
        src_mask = (words != constant.PAD_ID).unsqueeze(-2)

        word_embs = self.emb(words)
        embs = [word_embs]

        if self.opt['pos_dim'] > 0:
            embs += [self.pos_emb(pos)]
        embs = torch.cat(embs, dim=2)
        embs = self.in_drop(embs)

        if self.opt.get('rnn', False):
            embs = self.input_W_R(embs)
            gcn_inputs = self.rnn_drop(self.encode_with_rnn(embs, masks, words.size()[0]))
        else:
            gcn_inputs = embs

        gcn_inputs = self.input_W_G(gcn_inputs)

        layer_list = []
        outputs = gcn_inputs

        adj_list = None

        for i in range(len(self.layers)):
            if i == 0 or i == 3:
                adj_list = self.layers[i](outputs, src_mask)
                if self.opt['data_dir'] != 'dataset/semeval':
                    for j in range(len(adj_list)):
                        if i == 3:
                            adj_list[j] = entmax_bisect(adj_list[j], self.alpha_list[self.heads + j])
                        else:
                            adj_list[j] = entmax_bisect(adj_list[j], self.alpha_list[j])
            else:
                outputs = self.layers[i](adj_list, outputs)
                layer_list.append(outputs)

        aggregate_out = torch.cat(layer_list, dim=2)
        dcgcn_output = self.aggregate_W(aggregate_out)
        adj = torch.stack(adj_list, dim=1).sum(dim=1)

        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)

        return dcgcn_output, mask


class MultiGraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, opt, mem_dim, layers, heads):
        super(MultiGraphConvLayer, self).__init__()
        self.opt = opt
        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.heads = heads
        self.gcn_drop = nn.Dropout(opt['gcn_dropout'])

        # dcgcn layer
        self.Linear = nn.Linear(self.mem_dim * self.heads, self.mem_dim)
        self.weight_list = nn.ModuleList()

        for i in range(self.heads):
            for j in range(self.layers):
                self.weight_list.append(nn.Linear(self.mem_dim + self.head_dim * j, self.head_dim))

        self.weight_list = self.weight_list.cuda()
        self.Linear = self.Linear.cuda()

    def forward(self, adj_list, gcn_inputs):

        multi_head_list = []
        for i in range(self.heads):
            adj = adj_list[i]
            denom = adj.sum(2).unsqueeze(2) + 1
            outputs = gcn_inputs
            cache_list = [outputs]
            output_list = []
            for l in range(self.layers):
                index = i * self.layers + l
                Ax = adj.bmm(outputs)
                AxW = self.weight_list[index](Ax)
                AxW = AxW + self.weight_list[index](outputs)  # self loop
                AxW = AxW / denom
                gAxW = F.relu(AxW)
                cache_list.append(gAxW)
                outputs = torch.cat(cache_list, dim=2)
                output_list.append(self.gcn_drop(gAxW))

            gcn_ouputs = torch.cat(output_list, dim=2)
            gcn_ouputs = gcn_ouputs + gcn_inputs

            multi_head_list.append(gcn_ouputs)

        final_output = torch.cat(multi_head_list, dim=2)
        out = self.Linear(final_output)

        return out


class StructuredAttention(nn.Module):
    def __init__(self, sent_hiddent_size, trees_num):#, py_version):
        super(StructuredAttention, self).__init__()

        self.str_dim_size = sent_hiddent_size #- self.sem_dim_size

        self.model_dim = sent_hiddent_size

        self.linear_roots = nn.ModuleList()

        self.trees_num = trees_num

        for i in range(self.trees_num):
            self.linear_roots.append(nn.Linear(self.model_dim, 1))

        self.attn = MultiHeadAttention(self.trees_num, self.model_dim)

    def forward(self, input, src_mask):  # batch*sent * token * hidden
        batch_size, token_size, dim_size = input.size()

        attn_tensor = self.attn(input, input, src_mask)
        attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]

        adj_list = list()

        for i in range(self.trees_num):
            f_i = self.linear_roots[i](input).squeeze(-1)
            f_ij = attn_adj_list[i]

            mask = torch.ones(f_ij.size(1), f_ij.size(1)) - torch.eye(f_ij.size(1), f_ij.size(1))
            mask = mask.unsqueeze(0).expand(f_ij.size(0), mask.size(0), mask.size(1)).cuda()
            A_ij = torch.exp(f_ij) * mask

            tmp = torch.sum(A_ij, dim=1)  # nan: dimension
            res = torch.zeros(batch_size, token_size, token_size).cuda()

            res.as_strided(tmp.size(), [res.stride(0), res.size(2) + 1]).copy_(tmp)
            L_ij = -A_ij + res  # A_ij has 0s as diagonals

            L_ij_bar = L_ij
            L_ij_bar[:, 0, :] = f_i

            LLinv = torch.inverse(L_ij_bar)

            d0 = f_i * LLinv[:, :, 0]

            LLinv_diag = torch.diagonal(LLinv, dim1=-2, dim2=-1).unsqueeze(2)

            tmp1 = (A_ij.transpose(1, 2) * LLinv_diag).transpose(1, 2)
            tmp2 = A_ij * LLinv.transpose(1, 2)

            temp11 = torch.zeros(batch_size, token_size, 1)
            temp21 = torch.zeros(batch_size, 1, token_size)

            temp12 = torch.ones(batch_size, token_size, token_size - 1)
            temp22 = torch.ones(batch_size, token_size - 1, token_size)

            mask1 = torch.cat([temp11, temp12], 2).cuda()
            mask2 = torch.cat([temp21, temp22], 1).cuda()

            dx = mask1 * tmp1 - mask2 * tmp2

            d = torch.cat([d0.unsqueeze(1), dx], dim=1)
            df = d.transpose(1, 2)

            att = df[:, :, 1:]

            adj_list.append(att)

        return adj_list


def pool(h, mask, type='max'):
    if type == 'max':
        h = h.masked_fill(mask, -constant.INFINITY_NUMBER)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)


def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    return h0.cuda(), c0.cuda()


def attention(query, key, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)

        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key))]
        attn = attention(query, key, mask=mask, dropout=self.dropout)

        return attn