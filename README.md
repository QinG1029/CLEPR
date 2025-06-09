# Contrastive Learning Enhanced Probabilistic Label Refinement (CLEPR)

## Installation

```shell
conda create -n clepr python=3.7 -y
conda activate clepr
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch -y
conda install -c pytorch -c nvidia faiss-gpu -y

git clone https://github.com/xieincz/CLEPR.git
cd CLEPR
pip install -r requirements.txt
```

## Prepare Datasets

```shell
cd examples && mkdir data
```
Download the raw datasets [DukeMTMC-reID](https://arxiv.org/abs/1609.01775), [Market-1501](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf), [MSMT17](https://arxiv.org/abs/1711.08565),
and then unzip them under the directory like
```
MMT/examples/data
├── dukemtmc
│   └── DukeMTMC-reID
├── market1501
│   └── Market-1501-v15.09.15
└── msmt17
    └── MSMT17_V1
```

## Custom Datasets
Change Line 24 of clepr/datasets/custom.py to the path of your_custom_dataset. If your have multiple custom datasets, you can copy and rewrite clepr/datasets/custom.py according to your data.
```
MMT/examples/data
├── dukemtmc
│   └── DukeMTMC-reID
├── market1501
│   └── Market-1501-v15.09.15
└── custom
    └── your_custom_dataset
        |── trianval
        |── probe
        └── gallery
    
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
**Market-to-MSMT (ResNet-50)**
```shell
# pre-training on the source domain
sh scripts/pretrain_single_gpu.sh market1501 msmt17 resnet50 1
sh scripts/pretrain_single_gpu.sh market1501 msmt17 resnet50 2
# end-to-end training with CLEPR
sh scripts/train_CLEPR_single_gpu.sh market1501 msmt17 resnet50 1500 0.3 0.1 400 120
# testing the best model
sh scripts/test.sh msmt17 resnet logs/market1501TOmsmt17/resnet-CLEPR-3000-0.3/model_best.pth.tar
```
**Duke-to-MSMT (ResNet-50)**
```shell
# pre-training on the source domain
sh scripts/pretrain_single_gpu.sh dukemtmc msmt17 resnet50 1
sh scripts/pretrain_single_gpu.sh dukemtmc msmt17 resnet50 2
# end-to-end training with CLEPR
sh scripts/train_CLEPR_single_gpu.sh dukemtmc msmt17 resnet50 1500 0.3 0.1 400 120
# testing the best model
sh scripts/test.sh msmt17 resnet logs/dukemtmcTOmsmt17/resnet-CLEPR-3000-0.3/model_best.pth.tar
```



## Reported Results
*The reported results of this repo on four main-stream UDA Re-ID benchmarks are listed below.*
![results](figs/results.PNG)



|      | dukemtmc TO market1501 | market1501 TO dukemtmc | market1501 TO msmt17 | dukemtmc TO msmt17 |
| ---- | ---------------------- | ---------------------- | -------------------- | ------------------ |
| mAP  | 75.4                   | 68                     | 23.5                 | 25.5（94epoch）    |
| R1   | 88.9                   | 80.3                   | 48.8                 |                    |
| R5   | 95.3                   | 89.5                   | 62.3                 |                    |
| R10  | 96.9                   | 92.9                   | 68.1                 |                    |

单卡



## Acknowledgment

Thanks to JeyesHan and TrinhQuocNguyen for their excellent work. Most of the code in this work was borrowed from them.



## Citation

If you find this code useful for your research, please cite our paper
```




```
