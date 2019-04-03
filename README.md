# Destruction and Construction Learning for Fine-grained Image Recognition

By Yue Chen, Yalong Bai, Wei Zhang, Tao Mei

### Introduction

This code is relative to the [DCL](https://arxiv.org/), which is accepted on CVPR 2019.

This DCL code in this repo is written based on Pytorch 0.4.0.

This code has been tested on Ubuntu 16.04.3 LTS with Python 3.6.5 and CUDA 9.0.

Yuo can use this public docker image as the test environment:

```shell
docker pull pytorch/pytorch:0.4-cuda9-cudnn7-devel
```

### Citing DCL

If you find this repo useful in your research, please consider citing:

    @article{chen2019dcl,
      title={Destruction and Construction Learning for Fine-grained Image Recognition},
      author={Chen Yue and Bai, Yalong and Zhang Wei and Mei Tao},
      journal={arXiv preprint arXiv:},
      year={2019}
    }

### Requirements

0. Pytorch 0.4.0

0. Numpy, Pillow, Pandas

0. GPU: P40, etc. (May have bugs on the latest V100 GPU)

### Datasets Prepare

0. Download CUB-200-2011 dataset form [Caltech-UCSD Birds-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)

0. Unzip the dataset file under the folder 'datasets'

0. Run ./datasets/CUB_pre.py to generate annotation files 'train.txt', 'test.txt' and image folder 'all' for CUB-200-2011 dataset

### Testing Demo

0. Download `CUB_model.pth` from [Google Drive](https://drive.google.com/file/d/1xWMOi5hADm1xMUl5dDLeP6cfjZit6nQi/view?usp=sharing).

0. Run `CUB_test.py`

### Training on CUB-200-2011

0. Run `train.py` to train and test the CUB-200-2011 datasets. Wait about half day for training and testing.

0. Hopefully it would give the evaluation results around ~87.8% acc after running.

**Support for other datasets will be updated later**
