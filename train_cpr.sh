#!/bin/bash

CUDA_VISIBLE_DEVICES=1  python3 train.py --id cpr --seed 1 --vocab_dir dataset/cpr --data_dir dataset/cpr --hidden_dim 300 --lr 0.3 --rnn_hidden 300 --num_epoch 100 --pooling max --mlp_layers 1 --num_layers 2 --pooling_l2 0.002


