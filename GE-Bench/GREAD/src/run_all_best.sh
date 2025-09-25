#!/bin/bash

datasets=(Cora Citeseer Cornell Texas Wisconsin Chameleon Squirrel)
times=(2 4 8 16 32 64 128)
lrs=(0.01 0.005 0.001)

# 创建results目录和每个数据集的子目录
mkdir -p results
for dataset in "${datasets[@]}"; do
  mkdir -p "results/${dataset}_bestopt"
  mkdir -p "results/${dataset}_bestopt/folds_results"
done

for dataset in "${datasets[@]}"; do
  for time in "${times[@]}"; do
    echo "Searching best lr for $dataset, time=$time"
    python run_GNN_folds.py \
      --dataset $dataset \
      --time $time \
      --step_size 1 \
      --method rk4 \
      --seed 12345 \
      --not_lcc \
      --hetero_undir \
      > results/${dataset}_bestopt/final_time${time}.log 2>&1
  done
done