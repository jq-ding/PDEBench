#!/bin/bash

# 定义数据集和参数
datasets=(Cora Citeseer Chameleon Squirrel)
# datasets=(Squirrel)
layers=(2 4 8 16 32 64 128)
lrs=(0.01 0.005 0.001)

# 创建results目录和每个数据集的子目录
mkdir -p results
for dataset in "${datasets[@]}"; do
  mkdir -p "results/${dataset}_bestopt"
  mkdir -p "results/${dataset}_bestopt/folds_results"
done

# 对每个数据集和层数进行实验
for dataset in "${datasets[@]}"; do
  for num_layers in "${layers[@]}"; do
    echo "Running $dataset with $num_layers layers"
    python node_classification_folds.py \
      --dataset $dataset \
      --num_layers $num_layers \
      --num_split 10 \
      --undirected \
      --sklearn \
      --best \
      > results/${dataset}_bestopt/final_layers${num_layers}.log 2>&1
  done
done 