# Latent-Forests
Learning Latent Forests for Medical Relation Extraction (authors' PyTorch implementation for the IJCAI20 paper)

Learning Latent Forests for Medical Relation Extraction
==========

This paper/code introduces the Latent Forests Graph Convolutional graph convolutional networks (LFGCNs)  for the medical relation extraction task.

You can find the paper [here](https://www.ijcai.org/Proceedings/2020/0505.pdf)

 

## Requirements

Our model was trained on GPU GeForce RTX 2080 Ti P100-SXM2 of Ubuntu 16.04.6 LTS

- Python 3 (tested on 3.6.11)

- PyTorch (tested on 1.5.0)

- CUDA (tested on 10.2)

- tqdm

- unzip, wget (for downloading only)

We have released our trained model in this repo. You can find the trained model under the saved_models directory. There is no guarantee that the model is the same as we released and reported if you run the code on different environments (including hardware and software). 

## Preparation

The code includes three datasets: CPR, PGR and Semeval, all of them under the directory `dataset`.

  
First, download and unzip GloVe vectors:

```
chmod +x download.sh; ./download.sh
```

  

Then prepare vocabulary and initial word vectors for different datasets (cpr/pgr/semeval). Take CPR as an walking example. You can repalce the cpr with other datasets:

```
python3 prepare_vocab.py dataset/cpr dataset/cpr --glove_dir dataset/glove
```

  

This will write vocabulary and word vectors as a numpy matrix into their corresponding dir `dataset/cpr`.

  

## Training

  

To train the LFGCN model, run:

```
chmod +x train_cpr.sh; ./train_cpr.sh
```

  

Model checkpoints and logs will be saved to `./saved_models/cpr`.

  

For details on the use of other parameters, please refer to `train.py`.

  

## Evaluation

  

Our pretrained model is saved under the dir saved_models/cpr. To run evaluation on the test set, run:

```
python3 eval.py saved_model/cpr --data_dir dataset/cpr
```

  

Use `--model checkpoint_epoch_100.pt` to specify a model checkpoint file.


```

## Related Repo

The paper uses the model DCGCN, for detail architecture please refer to the TACL19 paper [Densely Connected Graph Convolutional Network for Graph-to-Sequence Learning](https://github.com/Cartus/DCGCN). Codes are adapted from the repo of the AGGCN paper [Attention Guided Graph Convolutional Networks for Relation Extraction](https://arxiv.org/pdf/1906.07510.pdf).

## Citation

```
@inproceedings{Guo2020LearningLF,
  title={Learning Latent Forests for Medical Relation Extraction},
  author={Zhijiang Guo and Guoshun Nan and Wei Lu and Shay B. Cohen},
  booktitle={Proc. of IJCAI},
  year={2020}
}
```

