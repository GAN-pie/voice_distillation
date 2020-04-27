#!/bin/bash

set -e

python bin/train-teacher.py data/lst_skyrim/train.lst data/lst_skyrim/val.lst data/skyrim.txt exp/ \
    --input-dim 400 --emb-dim 64 --drop-rate 0.25 --batch 12 --epochs 100 \
    --temperature 2.0 --soft-lst data/masseffect.lst --soft-features data/masseffect.txt

for k in `seq 1 4`
do

    python bin/train-student-dual.py data/lst_masseffect/train_${k}.lst data/lst_masseffect/val_${k}.lst data/masseffect.txt exp/masseffect_soft_targets.txt exp/pvector_${k} \
        --input-dim 400 --emb-dim 64 --temperature 2.0 --lambda 0.5 --epochs 100 --drop-rate 0.25

    python bin/embedding-extract.py data/masseffect.lst data/masseffect.txt exp/pvector_${k}

    python bin/vector-clustering.py data/lst_masseffect/test_${k}.lst exp/pvector_${k}/masseffect_pvectors.txt exp/pvector_${k}
done

