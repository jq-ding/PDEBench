#!/bin/bash

# 定义数据集和参数
datasets=(Cora Citeseer Cornell Texas Wisconsin Chameleon Squirrel)
# datasets=(Squirrel)
layers=(2 4 8 16 32 64 128)
lrs=(0.01 0.005 0.001)

# 创建results目录和每个数据集的子目录
mkdir -p results
for dataset in "${datasets[@]}"; do
  mkdir -p "results/${dataset}"
  mkdir -p "results/${dataset}/folds_results"
done

# 对每个数据集和层数进行实验
for dataset in "${datasets[@]}"; do
  for num_layers in "${layers[@]}"; do
    best_lr=""
    best_val_acc=-1
    echo "Searching best lr for $dataset, num_layers=$num_layers"
    
    # 在第一个split上搜索最佳学习率
    for lr in "${lrs[@]}"; do
      echo "Testing lr=$lr"
      python node_classification_folds.py \
        --dataset $dataset \
        --num_layers $num_layers \
        --learning_rate $lr \
        --split_id 0 \
        --num_split 1 \
        --num_epochs 1000 \
        --hidden_channels 64 \
        --normalize_features \
        --layer_norm \
        --no_sharing \
        --undirected \
        --sklearn \
        > results/${dataset}/search_layers${num_layers}_lr${lr}.log 2>&1

      # 解析验证集acc
      if command -v jq >/dev/null 2>&1; then
        val_acc=$(jq '.val.acc.val' results/${dataset}/single_split_summary_layers${num_layers}.json)
      else
        # 使用更精确的grep模式来匹配validation accuracy，确保只获取一个数字
        val_acc=$(grep -A 5 '"acc":' results/${dataset}/single_split_summary_layers${num_layers}.json | grep '"val":' | head -n 1 | grep -o '[0-9.]*' | head -n 1)
      fi

      if [ -z "$val_acc" ]; then
        echo "Warning: Could not find validation accuracy in results/${dataset}/single_split_summary_layers${num_layers}.json"
        continue
      fi

      echo "lr=$lr, val_acc=$val_acc"
      # 使用awk进行浮点数比较
      if [ $(echo "$val_acc $best_val_acc" | awk '{if ($1 > $2) print 1; else print 0}') -eq 1 ]; then
        best_val_acc=$val_acc
        best_lr=$lr
      fi
    done

    if [ -z "$best_lr" ]; then
      echo "Warning: No valid learning rate found for $dataset, num_layers=$num_layers"
      continue
    fi

    echo "Best lr for $dataset, num_layers=$num_layers: $best_lr (val_acc=$best_val_acc)"

    # 用最佳lr运行完整10折实验
    python node_classification_folds.py \
      --dataset $dataset \
      --num_layers $num_layers \
      --learning_rate $best_lr \
      --num_epochs 1000 \
      --hidden_channels 64 \
      --num_split 10 \
      --undirected \
      --normalize_features \
      --sklearn \
      --layer_norm \
      --no_sharing \
      > results/${dataset}/final_layers${num_layers}_lr${best_lr}.log 2>&1
  done
done 