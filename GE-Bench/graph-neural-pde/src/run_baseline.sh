#!/bin/bash

datasets=(Cora Citeseer Cornell Texas Wisconsin Chameleon Squirrel)
layers=(2 4 8 16 32 64 128)
lrs=(0.01 0.005 0.001)
models=(GCN GAT)

# 创建results目录和每个数据集的子目录
mkdir -p baseline_results
for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    mkdir -p "baseline_results/$model/${dataset}"
    mkdir -p "baseline_results/$model/${dataset}/folds_results"
  done
done

for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    for time in "${layers[@]}"; do
      best_lr=""
    best_val_acc=-1
    echo "Searching best lr for $dataset, time=$time"
    for lr in "${lrs[@]}"; do
      # 只在split=0上搜索
      python run_baseline_folds.py \
        --dataset $dataset \
        --num_layers $time \
        --model_type $model \
        --lr $lr \
        --split_id 0 \
        --epoch 1000 \
        --seed 12345 \
        --not_lcc \
        > baseline_results/${model}/${dataset}/search_time${time}_lr${lr}.log 2>&1

    #   # 调试信息：检查文件是否存在
    #   echo "Checking file: results/${dataset}/single_split_summary_time${time}.0.json"
    #   if [ -f "results/${dataset}/single_split_summary_time${time}.0.json" ]; then
    #     echo "File exists. Content:"
    #     cat "results/${dataset}/single_split_summary_time${time}.0.json"
    #   else
    #     echo "File does not exist!"
    #     # 列出目录内容
    #     echo "Directory contents:"
    #     ls -l "results/${dataset}/"
    #   fi

      # 解析验证集acc（从single_split_summary.json中获取）
      if command -v jq >/dev/null 2>&1; then
        val_acc=$(jq '.[0].val.acc' baseline_results/${model}/${dataset}/single_split_summary_time${time}.json)
      else
        val_acc=$(grep -o '"acc":[ ]*[0-9.]\+' baseline_results/${model}/${dataset}/single_split_summary_time${time}.json | head -1 | grep -o '[0-9.]*')
      fi
      if [ -z "$val_acc" ]; then
        echo "Warning: Could not find validation accuracy in baseline_results/${model}/${dataset}/single_split_summary_time${time}.json"
        continue
      fi
      echo "lr=$lr, val_acc=$val_acc"
      if (( $(echo "$val_acc > $best_val_acc" | bc -l) )); then
        best_val_acc=$val_acc
        best_lr=$lr
      fi
    done

    if [ -z "$best_lr" ]; then
      echo "Warning: No valid learning rate found for $dataset, time=$time"
      continue
    fi

    echo "Best lr for $dataset, time=$time: $best_lr (val_acc=$best_val_acc)"

    # 用最佳lr跑10折
    python run_baseline_folds.py \
      --dataset $dataset \
      --num_layers $time \
      --model_type $model \
      --lr $best_lr \
      --epoch 1000 \
      --seed 12345 \
      --not_lcc \
        > baseline_results/${model}/${dataset}/final_time${time}_lr${best_lr}.log 2>&1
    done
  done
done