#!/bin/bash

set -e

export CUDA_VISIBLE_DEVICES=0

# python bin/train-teacher.py data/lst_skyrim/train.lst data/lst_skyrim/val.lst data/xvector_skyrim.txt exp/ \
#     --input-dim 512 --emb-dim 64 --drop-rate 0.25 --batch 12 --epochs 100 \
#     --temperature 2.0 --soft-lst data/masseffect.lst --soft-features data/xvector_masseffect.txt

for k in `seq 1 4`
do
    if [ ! -d "exp/pvector_${k}" ]; then
        mkdir "exp/pvector_${k}"
    fi

    python bin/train-student-dual.py data/lst_masseffect/train_${k}.lst data/lst_masseffect/val_${k}.lst data/xvector_masseffect.txt \
        exp/xvector_masseffect_soft_targets.txt exp/pvector_${k} \
        --input-dim 512 --emb-dim 64 --temperature 2.0 --lambda 0.5 --epochs 100 --drop-rate 0.25

    python bin/embedding-extract.py data/masseffect.lst data/xvector_masseffect.txt exp/pvector_${k}

    python bin/vector-clustering.py data/lst_masseffect/test_${k}.lst exp/pvector_${k}/xvector_masseffect_pvectors.txt exp/pvector_${k}
done

