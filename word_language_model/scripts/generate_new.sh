#! /bin/bash

scripts=`dirname "$0"`
base=$(realpath $scripts/..)

models=$base/models
data=$base/data
tools=$base
samples=$base/samples

mkdir -p $samples

num_threads=12
device=0

(CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python generate.py \
        --data $data/positive \
        --words 100 \
        --checkpoint $models/model_pos.pt \
        --outf $samples/sample_pos_generate.txt \
		--input "A remarkable example of"
		
		
)