#!/bin/bash

gpu_id=0
export CUDA_VISIBLE_DEVICES=$gpu_id

# 定义要测试的模型列表
models=("meta-llama/Meta-Llama-3-8B" "meta-llama/Llama-2-7b-hf")

# 定义要测试的量化比特数
bits=(3)

# 双层循环：遍历所有模型和比特数组合
for model in "${models[@]}"; do
    for w_bit in "${bits[@]}"; do
        echo "========================================"
        echo "Running: Model=$model, Bits=$w_bit"
        echo "========================================"
        
        python main.py --model $model \
         --w_bits $w_bit \
         --w_groupsize 256 \
         --cal_dataset c4 \
         --a_bits 16 \
         --v_bits 16 \
         --k_bits 16 \
         --w_asym \
         --w_clip \
         --asym_calibrate \
         --bsz 1
         
        echo "Finished: Model=$model, Bits=$w_bit"
    done
done
#  --rotate \