#!/bin/bash

datasets=(Texas Wisconsin)
nlayers=(2)
# lrs=(0.01 0.005 0.001)

# 创建results目录和每个数据集的子目录
mkdir -p results
for dataset in "${datasets[@]}"; do
  mkdir -p "results/${dataset}_bestopt"
  mkdir -p "results/${dataset}_bestopt/folds_results"
done

for dataset in "${datasets[@]}"; do
  for nlayers in "${nlayers[@]}"; do
    echo "Searching best lr for $dataset, nlayers=$nlayers"
    python run_folds_heter.py \
      --dataset $dataset \
      --nlayers $nlayers \
      --n_splits 10 \
      --seed 12345 \
      > results/${dataset}_bestopt/final_nlayers${nlayers}.log 2>&1
  done
done