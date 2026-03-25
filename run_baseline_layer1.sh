#!/bin/bash

# 基准项目：只量化第一层并保存输出
# 用于与改进项目进行对比

gpu_id=0
export CUDA_VISIBLE_DEVICES=$gpu_id

MODEL="meta-llama/Llama-2-7b-hf"
# MODEL="meta-llama/Meta-Llama-3-8B"  # 可选 LLaMA3-8B

# ===== 有 Rotate 版本 =====
echo "===== Running Baseline: Layer 1 Only (With Rotate) ====="
python main.py --model $MODEL \
 --w_bits 3 \
 --w_groupsize 256 \
 --cal_dataset c4 \
 --a_bits 16 \
 --v_bits 16 \
 --k_bits 16 \
 --w_asym \
 --w_clip \
 --asym_calibrate \
 --act_weight_mode none \
 --rotate \
 --quant_first_layer_only \
 --save_layer1_output layer1_baseline_with_rotate_output.pt \
 --bsz 1

echo "===== Baseline experiments completed! ====="
echo ""
echo "Generated files:"
echo "  - layer1_baseline_with_rotate_output.pt"
echo "  - layer1_baseline_with_rotate_input.pt"
echo "  - layer1_baseline_with_rotate_output_info.txt"
