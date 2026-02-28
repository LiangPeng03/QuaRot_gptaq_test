#!/bin/bash

gpu_id=0,1
export CUDA_VISIBLE_DEVICES=$gpu_id

$HOME/.conda/envs/awq/bin/python main.py --model facebook/opt-125m \
 --w_bits 3 \
 --w_groupsize 256 \
 --cal_dataset c4 \
 --a_bits 16 \
 --v_bits 16 \
 --k_bits 16 \
 --w_asym \
 --w_clip \
 --enable_aq_calibration \
 --bsz 1 \