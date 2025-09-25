#!/bin/bash

datasets=(Citeseer Cornell Texas Wisconsin Chameleon Squirrel)
nlayers=(2)
lrs=(0.01 0.005 0.001)

# 创建results目录和每个数据集的子目录
mkdir -p results
for dataset in "${datasets[@]}"; do
  mkdir -p "results/${dataset}"
  mkdir -p "results/${dataset}/folds_results"
done

for dataset in "${datasets[@]}"; do
  for nlayers in "${nlayers[@]}"; do
    best_lr=""
    best_val_acc=-1
    echo "Searching best lr for $dataset, nlayers=$nlayers"
    for lr in "${lrs[@]}"; do
      # 只在split=0上搜索
      python main_nc_folds.py \
        --dataset $dataset \
        --num_layers $nlayers \
        --lr $lr \
        --split_id 0 \
        --epoch 1000 \
        --seed 12345 \
        --num_splits 1 \
        > results/${dataset}/search_nlayers${nlayers}_lr${lr}.log 2>&1


      # 解析验证集acc（从single_split_summary.json中获取）
      if command -v jq >/dev/null 2>&1; then
        val_acc=$(jq '.[0].val.acc' results/${dataset}/single_split_summary_nlayers${nlayers}.json)
      else
        val_acc=$(grep -o '"acc":[ ]*[0-9.]\+' results/${dataset}/single_split_summary_nlayers${nlayers}.json | head -1 | grep -o '[0-9.]*')
      fi
      if [ -z "$val_acc" ]; then
        echo "Warning: Could not find validation accuracy in results/${dataset}/single_split_summary_nlayers${nlayers}.json"
        continue
      fi
      echo "lr=$lr, val_acc=$val_acc"
      if (( $(echo "$val_acc > $best_val_acc" | bc -l) )); then
        best_val_acc=$val_acc
        best_lr=$lr
      fi
    done

    if [ -z "$best_lr" ]; then
      echo "Warning: No valid learning rate found for $dataset, nlayers=$nlayers"
      continue
    fi

    echo "Best lr for $dataset, nlayers=$nlayers: $best_lr (val_acc=$best_val_acc)"

    # 用最佳lr跑10折
    python main_nc_folds.py \
      --dataset $dataset \
      --num_layers $nlayers \
      --lr $best_lr \
      --epoch 1000 \
      --num_splits 10 \
      --seed 12345 \
      > results/${dataset}/final_nlayers${nlayers}_lr${best_lr}.log 2>&1
  done
done