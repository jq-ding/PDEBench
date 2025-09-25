#!/bin/bash

datasets=(Cora Citeseer)
times=(2 4 8 16 32 64 128)
mkdir -p results
for dataset in "${datasets[@]}"; do
  mkdir -p "results/${dataset}_bestopt"
  mkdir -p "results/${dataset}_bestopt/folds_results"
done

for dataset in "${datasets[@]}"; do
  for time in "${times[@]}"; do
    echo "Searching best lr for $dataset, time=$time"
    python run_folds_homo.py \
      --dataset $dataset \
      --time $time \
      --step_size 1 \
      --method rk4 \
      --seed 12345 \
      --no_early \
      --not_lcc \
      --augment \
      > results/${dataset}_bestopt/final_time${time}.log 2>&1
  done
done