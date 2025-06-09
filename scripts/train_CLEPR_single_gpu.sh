#!/bin/sh
SOURCE=$1
TARGET=$2
ARCH=$3
CLUSTER=$4
P=$5
TEMPERATURE=$6
ITERS=$7
EPOCHS=$8

if [ $# -ne 8 ];
  then
    echo "Arguments error: <SOURCE> <TARGET> <ARCH> <CLUSTER NUM> <P0> <TEMPERATURE> <ITERS> <EPOCHS>"
    exit 1
fi

#CUDA_VISIBLE_DEVICES=0,1,2,3 \
CUDA_VISIBLE_DEVICES=0 \
python examples/train_CLEPR.py -dt ${TARGET} -a ${ARCH} -j 16 --num-clusters ${CLUSTER} \
	--num-instances 4 --lr 0.00035 --iters ${ITERS} -b 64 --epochs ${EPOCHS} --p ${P} \
	--soft-ce-weight 0.5 --soft-tri-weight 0.8 --dropout 0.5 --multiple_kmeans \
	--init-1 logs/${SOURCE}TO${TARGET}/${ARCH}-pretrain-1/model_best.pth.tar \
	--init-2 logs/${SOURCE}TO${TARGET}/${ARCH}-pretrain-2/model_best.pth.tar \
	--contrast_weight 0.1 --contrastive_learning_temperature ${TEMPERATURE} \
	--logs-dir logs/${SOURCE}TO${TARGET}/${ARCH}-CLEPR-${CLUSTER}-${P}-T${TEMPERATURE}-iters${ITERS}-epochs${EPOCHS}
