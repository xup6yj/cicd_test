# -*- coding: utf-8 -*-
import os
import csv
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import einsum
from torch.utils.data import DataLoader
import torchvision.transforms.functional as tv_F

import torchio as tio

from utils import *

if torch.cuda.is_available:
    device = 'cuda'
    torch.backends.cudnn.benchmark = True
else:
    device = 'cpu'


def measurement(outputs, targets):
    '''
    :param: outputs: (b, c, x, y(, z))
    :param: targets: (b, 1, x, y(, z))
    :return: acc, iou, tpr, tnr, fpr, fnr, dsc, ppv, tp, tn, fp, fn
    '''
    num_samples = outputs.shape[0]
    outputs_shape = outputs.shape
    targets_shape = targets.shape

    if len(outputs_shape) != len(targets_shape):
        targets = targets.view((targets_shape[0], 1, *targets_shape[1:]))

    outputs_one_ch = outputs[:,-1:,...]
    tp = outputs_one_ch * targets
    tn = (1-outputs_one_ch) * (1-targets)
    fp = outputs_one_ch * (1-targets)
    fn = (1-outputs_one_ch) * targets

    tp = einsum('bcxyz->b', tp).type(torch.float32)
    tn = einsum('bcxyz->b', tn).type(torch.float32)
    fp = einsum('bcxyz->b', fp).type(torch.float32)
    fn = einsum('bcxyz->b', fn).type(torch.float32)

    acc = (tp+tn) / (tp+tn+fp+fn)

    iou = torch.zeros((num_samples, 1))
    dsc = torch.zeros((num_samples, 1))
    tnr = torch.zeros((num_samples, 1))
    fpr = torch.zeros((num_samples, 1))
    tpr = torch.zeros((num_samples, 1))
    fnr = torch.zeros((num_samples, 1))
    ppv = torch.zeros((num_samples, 1))
    npv = torch.zeros((num_samples, 1))

    dsc = (2*tp) / (2*tp+fp+fn)
    iou = tp / (tp+fp+fn)
    for i in range(num_samples):
        if tn[i]+fp[i] != 0:
            tnr[i] = tn[i] / (tn[i]+fp[i])  # TNR = 1 - FPR, specificity, selectivity or true negative rate (TNR)
            fpr[i] = fp[i] / (fp[i]+tn[i])  # fall-out or false positive rate (FPR or false alarm rate)

        if tp[i]+fn[i] != 0:
            tpr[i] = tp[i] / (tp[i]+fn[i])  # TPR = 1 - FNR, sensitivity, recall, hit rate, or true positive rate (TPR)
            fnr[i] = fn[i] / (fn[i]+tp[i])  # miss rate or false negative rate (FNR)

        if tp[i]+fp[i] != 0:
            ppv[i] = tp[i] / (tp[i]+fp[i])  # precision or positive predictive value (PPV)

        if tn[i]+fn[i] != 0:
            npv[i] = tn[i] / (tn[i]+fn[i])  # negatie predictive value

    measure = {
        'acc': acc.mean().cpu().item(),
        'iou': iou.mean().cpu().item(),
        'tpr': tpr.mean().cpu().item(),
        'tnr': tnr.mean().cpu().item(),
        'fpr': fpr.mean().cpu().item(),
        'fnr': fnr.mean().cpu().item(),
        'dsc': dsc.mean().cpu().item(),
        'ppv': ppv.mean().cpu().item(),
        'npv': npv.mean().cpu().item(),
        'tp': tp.mean().cpu().item(),
        'tn': tn.mean().cpu().item(),
        'fp': fp.mean().cpu().item(),
        'fn': fn.mean().cpu().item()
    }
    return measure
