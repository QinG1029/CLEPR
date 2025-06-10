#Contrastive Learning Enhanced Pseudo-Labeling for Unsupervised Domain Adaptation in Person Re-identification(CLEPR)

## Installation

```shell
conda create -n clepr python=3.7 -y
conda activate clepr
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch -y
conda install -c pytorch -c nvidia faiss-gpu -y

git clone https://github.com/QinG1029/CLEPR
cd CLEPR
pip install -r requirements.txt
```

## Prepare Datasets

```shell
cd examples && mkdir data
```
Download the raw datasets [DukeMTMC-reID], [Market-1501], [CUHK03], [PersonX]
and then unzip them under the directory like
```
MMT/examples/data
├── dukemtmc
│   └── DukeMTMC-reID
├── market1501
│   └── Market-1501-v15.09.15
├── custom
│   └── CustomData
└── cuhk03
    └── CUHK03
```



## Example #1:

Transferring from [DukeMTMC-reID](https://arxiv.org/abs/1609.01775) to [Market-1501](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf) on the backbone of [ResNet-50](https://arxiv.org/abs/1512.03385), *i.e. Duke-to-Market (ResNet-50)*.

### Train
~~We utilize 4 RTX 3080x2 20GB GPUs for training.~~ (你可以将下面的命令中的 `_single_gpu` 删掉来用4个GPU进行训练)



We utilize 1 TITAN Xp GPU for training.



#### Stage I: Pre-training on the source domain

```shell
sh scripts/pretrain_single_gpu.sh dukemtmc market1501 resnet50 1
sh scripts/pretrain_single_gpu.sh dukemtmc market1501 resnet50 2
```

#### Stage II: End-to-end training with CLEPR
We utilized K-Means clustering algorithm in the paper.

```shell
sh scripts/train_CLEPR_single_gpu.sh dukemtmc market1501 resnet50 700 0.3 0.1 400 120
```

### Test
We utilize 1 GPU for testing.
Test the trained model with best performance by
```shell
sh scripts/test.sh market1501 resnet50 logs/dukemtmcTOmarket1501/resnet-CLEPR-700-0.3/model_best.pth.tar
```



## Other Examples:
**Market-to-Duke (ResNet-50)**
```shell
# pre-training on the source domain
sh scripts/pretrain_single_gpu.sh market1501 dukemtmc resnet50 1
sh scripts/pretrain_single_gpu.sh market1501 dukemtmc resnet50 2
# end-to-end training with CLEPR
sh scripts/train_CLEPR_single_gpu.sh market1501 dukemtmc resnet50 700 0.2 0.12 400 120
# testing the best model
sh scripts/test.sh dukemtmc resnet logs/market1501TOdukemtmc/resnet-CLEPR-700-0.2/model_best.pth.tar
```
**Market-to-CUHK (ResNet-50)**
```shell
# pre-training on the source domain
sh scripts/pretrain_single_gpu.sh market1501 cuhk03 resnet50 1
sh scripts/pretrain_single_gpu.sh market1501 cuhk03 resnet50 2
# end-to-end training with CLEPR
sh scripts/train_CLEPR_single_gpu.sh market1501 cuhk03 resnet50 700 0.3 0.1 400 120
# testing the best model
sh scripts/test.sh cuhk03 resnet logs/market1501TOcuhk03/resnet-CLEPR-3000-0.3/model_best.pth.tar
```
**Duke-to-CUHK (ResNet-50)**
```shell
# pre-training on the source domain
sh scripts/pretrain_single_gpu.sh dukemtmc cuhk03 resnet50 1
sh scripts/pretrain_single_gpu.sh dukemtmc cuhk03 resnet50 2
# end-to-end training with CLEPR
sh scripts/train_CLEPR_single_gpu.sh dukemtmc cuhk03 resnet50 700 0.3 0.1 400 120
# testing the best model
sh scripts/test.sh cuhk03 resnet logs/dukemtmcTOcuhk03/resnet-CLEPR-3000-0.3/model_best.pth.tar
```
**CUHK-to-Duke (ResNet-50)**
```shell
# pre-training on the source domain
sh scripts/pretrain_single_gpu.sh cuhk03 dukemtmc resnet50 1
sh scripts/pretrain_single_gpu.sh cuhk03 dukemtmc resnet50 2
# end-to-end training with CLEPR
sh scripts/train_CLEPR_single_gpu.sh cuhk03 dukemtmc resnet50 700 0.2 0.12 400 120
# testing the best model
sh scripts/test.sh dukemtmc resnet logs/cuhk03TOdukemtmc/resnet-CLEPR-3000-0.3/model_best.pth.tar
```
**CUHK-to-Market (ResNet-50)**
```shell
# pre-training on the source domain
sh scripts/pretrain_single_gpu.sh cuhk03 market1501 resnet50 1
sh scripts/pretrain_single_gpu.sh cuhk03 market1501 resnet50 2
# end-to-end training with CLEPR
sh scripts/train_CLEPR_single_gpu.sh cuhk03 market1501 resnet50 700 0.3 0.1 400 120
# testing the best model
sh scripts/test.sh market1501 resnet logs/cuhk03TOmarket1501/resnet-CLEPR-3000-0.3/model_best.pth.tar
```





## Reported Results
*The reported results of this repo on four main-stream UDA Re-ID benchmarks are listed below.*
![results](figs/results.PNG)



|      | dukemtmc TO market1501 | market1501 TO dukemtmc 
| ---- | ---------------------- | ---------------------- 
| mAP  | 79.0                   | 67.9                   
| R1   | 91.4                   | 81.4                                   
| R5   | 96.2                   | 89.7                                   
| R10  | 97.5                   | 92.3                                









```
