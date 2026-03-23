#!/bin/bash

# 3-day_gt-1_5fold

python inference.py \
--in_channel 1 --out_channel 2 --init_ch 32 --patch_x 256 --patch_y 256 --patch_z 8 --patch_overlap_x 0 --patch_overlap_y 0 --patch_overlap_z 2 --dataset ncct111_gt-1_5fold \
--model \
--cgm \
--cv 2 \
--cpt_path  \
--output_dir