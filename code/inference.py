# -*- coding: utf-8 -*-
import json
import warnings
from argparse import ArgumentParser

import torch
torch.cuda.set_device('cuda:0')

from modules import test3d

if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=DeprecationWarning) 

    parser = ArgumentParser()
    # for model
    parser.add_argument('--model', type=str, required=False, default='')
    parser.add_argument('--in_channel', type=int, default=1)
    parser.add_argument('--out_channel', type=int, default=2)
    parser.add_argument('--init_ch', type=int, default=32)
    parser.add_argument('--deconv', action='store_true', default=False)

    # for UNet3P
    parser.add_argument('--reduce_ch', type=int, default=64)

    # for RUNet, R2UNet
    parser.add_argument('--num_rcnn', type=int, default=2)
    parser.add_argument('--t', type=int, default=2)

    # for deep supervision
    parser.add_argument('--ds', action='store_true', default=False)

    # for classification guided module (multi-task)
    parser.add_argument('--cgm', action='store_true', default=True)

    # for dataloader
    parser.add_argument('--dataset', type=str, required=False, default='')
    parser.add_argument('--cv', type=int, required=False, default=3)
    parser.add_argument('--patch_x', type=int, required=False, default=256)
    parser.add_argument('--patch_y', type=int, required=False, default=256)
    parser.add_argument('--patch_z', type=int, required=False, default=8)
    parser.add_argument('--patch_overlap_x', type=int, required=False, default=0)
    parser.add_argument('--patch_overlap_y', type=int, required=False, default=0)
    parser.add_argument('--patch_overlap_z', type=int, required=False, default=2)

    parser.add_argument('--cpt_path', type=str, required=False, default='')
    parser.add_argument('--output_dir', type=str, required=False, default='')

    parser.add_argument('--test_time_aug', action='store_true', default=False)
    parser.add_argument('--test_time_aug_n', type=int, default=20)

    args = parser.parse_args()
    print(json.dumps(vars(args), indent=2))

    test3d(args)
