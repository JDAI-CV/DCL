[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/destruction-and-construction-learning-for/fine-grained-image-classification-on-fgvc)](https://paperswithcode.com/sota/fine-grained-image-classification-on-fgvc?p=destruction-and-construction-learning-for)
# Destruction and Construction Learning for Fine-grained Image Recognition

By Yue Chen, Yalong Bai, Wei Zhang, Tao Mei

Special thanks to [Yuanzhi Liang](https://github.com/akira-l) for code refactoring.

## UPDATE Jun. 21

Our solution for the FGVC Challenge 2019 (The Sixth Workshop on Fine-Grained Visual Categorization in CVPR 2019) is updated!

With ensemble of several DCL based classification models, we won:

- **First Place** in [iMaterialist Challenge on Product Recognition](https://www.kaggle.com/c/imaterialist-product-2019/leaderboard)
- **First Place** in [Fieldguide Challenge: Moths & Butterflies](https://www.kaggle.com/c/fieldguide-challenge-moths-and-butterflies/leaderboard)
- **Second Place** in [iFood - 2019 at FGVC6](https://www.kaggle.com/c/ifood-2019-fgvc6/leaderboard)

## Introduction

This project is a DCL pytorch implementation of [*Destruction and Construction Learning for Fine-grained Image Recognition*](http://openaccess.thecvf.com/content_CVPR_2019/html/Chen_Destruction_and_Construction_Learning_for_Fine-Grained_Image_Recognition_CVPR_2019_paper.html), CVPR2019. 

## Requirements

1. Python 3.6

2. Pytorch 0.4.0 or 0.4.1

3. CUDA 8.0 or higher

For docker environment:

```shell
docker pull pytorch/pytorch:0.4-cuda9-cudnn7-devel
```

For conda environment:

```shell
conda create --name DCL file conda_list.txt
```

For more backbone supports in DCL, please check [pretrainmodels](https://github.com/Cadene/pretrained-models.pytorch) and install:

```shell
pip install pretrainedmodels
```


## Datasets Prepare

1. Download correspond dataset to folder 'datasets'

2. Data organization: eg. CUB

     All the image data are in './datasets/CUB/data/'
    e.g. './datasets/CUB/data/*.jpg'

    The annotation files are in './datasets/CUB/anno/'
    e.g. './dataset/CUB/data/train.txt'

    In annotations:

    ```shell
    name_of_image.jpg label_num\n
    ```

    e.g. for CUB in repository:

    ```shell
    Black_Footed_Albatross_0009_34.jpg 0
    Black_Footed_Albatross_0014_89.jpg 0
    Laysan_Albatross_0044_784.jpg 1
    Sooty_Albatross_0021_796339.jpg 2
    ...
    ```

Some examples of datasets like CUB, Stanford Car, etc. are already given in our repository. You can use DCL to your datasets by simply converting annotations to train.txt/val.txt/test.txt and modify the class number in `config.py` as in line67: numcls=200.

## Training

Run `train.py` to train DCL.

For training CUB / STCAR / AIR from scratch

```shell
python train.py --data CUB --epoch 360 --backbone resnet50 \
                    --tb 16 --tnw 16 --vb 512 --vnw 16 \
                    --lr 0.0008 --lr_step 60 \
                    --cls_lr_ratio 10 --start_epoch 0 \
                    --detail training_descibe --size 512 \
                    --crop 448 --cls_mul --swap_num 7 7
```

For training CUB / STCAR / AIR from trained checkpoint

```shell
python train.py --data CUB --epoch 360 --backbone resnet50 \
                    --tb 16 --tnw 16 --vb 512 --vnw 16 \
                    --lr 0.0008 --lr_step 60 \
                    --cls_lr_ratio 10 --start_epoch $LAST_EPOCH \
                    --detail training_descibe4checkpoint --size 512 \
                    --crop 448 --cls_mul --swap_num 7 7
```

For training FGVC product datasets from scratch

```shell
 python train.py --data product --epoch 60 --backbone senet154 \
                    --tb 96 --tnw 32 --vb 512 --vnw 32 \
                    --lr 0.01 --lr_step 12 \
                    --cls_lr_ratio 10 --start_epoch 0 \
                    --detail training_descibe --size 512 \
                    --crop 448 --cls_2 --swap_num 7 7
```

For training FGVC datasets from trained checkpoint

```shell
 python train.py --data product --epoch 60 --backbone senet154 \
                    --tb 96 --tnw 32 --vb 512 --vnw 32 \
                    --lr 0.01 --lr_step 12 \
                    --cls_lr_ratio 10 --start_epoch $LAST_EPOCH \
                    --detail training_descibe4checkpoint --size 512 \
                    --crop 448 --cls_2 --swap_num 7 7
```
To achieve the similar results of paper, please use the default parameter settings.

## Citation
Please cite our CVPR19 paper if you use DCL in your work:
```
@InProceedings{Chen_2019_CVPR,
author = {Chen, Yue and Bai, Yalong and Zhang, Wei and Mei, Tao},
title = {Destruction and Construction Learning for Fine-Grained Image Recognition},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```


