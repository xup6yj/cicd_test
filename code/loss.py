# -*- coding: utf-8 -*-
import numpy as np

import torch
import torch.nn as nn
from torch import einsum

epsilon = 1e-7


# https://github.com/JunMa11/SegLoss/blob/master/test/loss_functions/dice_loss.py
def get_tp_fp_fn(outputs, targets):
    outputs_shape = outputs.shape
    targets_shape = targets.shape

    with torch.no_grad():
        if len(outputs_shape) != len(targets_shape):
            targets = targets.view((targets_shape[0], 1, *targets_shape[1:]))
        if all([i==j for i, j in zip(outputs.shape, targets.shape)]):
            targets_onehot = targets
        else:
            targets = targets.long()
            targets_onehot = torch.zeros(outputs_shape)
            if outputs.device.type == 'cuda':
                targets_onehot = targets_onehot.cuda(outputs.device.index)
            targets_onehot.scatter_(1, targets, 1)

    tp = outputs * targets_onehot
    fp = outputs * (1-targets_onehot)
    fn = (1-outputs) * targets_onehot
    tp = tp.sum(dim=(2, 3, 4))
    fp = fp.sum(dim=(2, 3, 4))
    fn = fn.sum(dim=(2, 3, 4))
    return tp, fp, fn


# https://github.com/JunMa11/SegLoss/blob/master/test/loss_functions/dice_loss.py
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1):
        '''
        paper: https://arxiv.org/pdf/1706.05721.pdf
        '''
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        tp, fp, fn = get_tp_fp_fn(inputs, targets)
        tversky = (tp+self.smooth) / (tp+self.alpha*fn+self.beta*fp+self.smooth)
        # tversky = tversky.mean()
        tversky = tversky[:,-1].mean()
        return -tversky


# https://github.com/JunMa11/SegLoss/blob/master/test/loss_functions/dice_loss.py
class FocalTverskyLoss(nn.Module):
    '''
    paper: https://arxiv.org/pdf/1810.07842.pdf
    author code: https://github.com/nabsabraham/focal-tversky-unet/blob/347d39117c24540400dfe80d106d2fb06d2b99e1/losses.py#L65
    '''
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75):
        super(FocalTverskyLoss, self).__init__()
        self.gamma = gamma
        self.tversky = TverskyLoss(alpha=alpha, beta=beta)

    def forward(self, inputs, targets):
        tversky_loss = 1 + self.tversky(inputs, targets)
        focal_tversky = torch.pow(tversky_loss, self.gamma)
        return focal_tversky


class DiceFocalLoss(nn.Module):
    def __init__(self, general_dice=True):
        super(DiceFocalLoss, self).__init__()
        self.focal_loss = FocalLoss()
        if general_dice:
            self.dice_loss = GDiceLoss()
        else:
            self.dice_loss = SDiceLoss()

    def forward(self, inputs, targets):
        focal_loss = self.focal_loss(inputs, targets)
        dice_loss = self.dice_loss(inputs, targets)
        return 0.5*focal_loss + 0.5*dice_loss


# https://github.com/JunMa11/SegLoss/blob/master/losses_pytorch/focal_loss.py
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0, 1]')

    def forward(self, inputs, targets):
        num_class = inputs.shape[1]

        if inputs.dim() > 2:
            # N, C, x, y, ... -> C, m (m = N * x * y * ...)
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
            inputs = inputs.permute(0, 2, 1).contiguous()
            inputs = inputs.view(-1, inputs.size(-1))
        targets = torch.squeeze(targets, 1)
        targets = targets.view(-1, 1)

        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1-self.alpha)
            alpha[self.balance_index] = self.alpha
        else:
            raise TypeError('Not support alpha type')

        if alpha.device != inputs.device:
            alpha = alpha.to(inputs.device)

        idx = targets.cpu().long()

        one_hot_key = torch.FloatTensor(targets.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != inputs.device:
            one_hot_key = one_hot_key.to(inputs.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth/(num_class-1), 1.0-self.smooth)
        pt = (one_hot_key*inputs).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1-pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


# https://github.com/JunMa11/SegLoss/blob/master/losses_pytorch/dice_loss.py
# generalized dice loss
class GDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(GDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs_shape = inputs.shape  # (batch size, class_num, x, y, ...)
        targets_shape = targets.shape  # (batch size, 1, x, y, ...)
        with torch.no_grad():
            if len(inputs_shape) != len(targets_shape):
                targets = targets.view((targets_shape[0], 1, *targets_shape[1:]))
            if all([i==j for i, j in zip(inputs.shape, targets.shape)]):
                targets_onehot = targets
            else:
                targets = targets.long()
                targets_onehot = torch.zeros(inputs_shape)
                if inputs.device.type == 'cuda':
                    targets_onehot = targets_onehot.cuda(inputs.device.index)
                targets_onehot.scatter_(1, targets, 1)

        class_weights = 1 / (einsum('bcxyz->bc', targets_onehot).type(torch.float32)+1e-10)**2
        intersection = class_weights * einsum('bcxyz, bcxyz->bc', inputs, targets_onehot)
        union = class_weights * (einsum('bcxyz->bc', inputs)+einsum('bcxyz->bc', targets_onehot))
        gdc = 2 * (einsum('bc->b', intersection)+self.smooth) / (einsum('bc->b', union)+self.smooth)
        return  1 - gdc.mean()


# https://github.com/JunMa11/SegLoss/blob/master/losses_pytorch/dice_loss.py
# soft dice loss
class SDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(SDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        tp, fp, fn = get_tp_fp_fn(inputs, targets)
        dc = (2*tp+self.smooth) / (2*tp+fp+fn+self.smooth)
        return 1 - dc.mean()


# part of unified focal loss
class AsymmetricFocalLoss(nn.Module):
    def __init__(self, delta=0.25, gamma=2):
        super(AsymmetricFocalLoss, self).__init__()
        '''For Imbalanced datasets
        Parameters
        ----------
        delta : float, optional
            controls weight given to false positive and false negatives, by default 0.25
        gamma : float, optional
            Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, by default 2
        '''
        self.delta = delta
        self.gamma = gamma

    def forward(self, inputs, targets):
        assert inputs.shape[1] > 1, 'The shape of model\'s output must be (b, c, h, w, ...)'
        inputs_shape = inputs.shape  # (b, c, h, w, ...)
        targets_shape = targets.shape  # (b, 1, h, w, ...)
        with torch.no_grad():
            if len(inputs_shape) != len(targets_shape):
                targets = targets.view((targets_shape[0], 1, *targets_shape[1:]))
            if all([i==j for i, j in zip(inputs.shape, targets.shape)]):
                targets_onehot = targets
            else:
                targets = targets.long()
                targets_onehot = torch.zeros(inputs_shape).to(inputs.device)
                targets_onehot.scatter_(1, targets, 1)

        inputs = torch.clamp(inputs, min=epsilon, max=1-epsilon)
        cross_entropy = -targets_onehot * torch.log(inputs)

        back_ce = torch.pow(1-inputs[:,0,...], self.gamma) * cross_entropy[:,0,...]
        back_ce =  (1-self.delta) * back_ce
        fore_ce = cross_entropy[:,1,...]
        fore_ce = self.delta * fore_ce

        loss = torch.stack((back_ce, fore_ce), dim=1).mean()
        return loss


# part of unified focal loss
class AsymmetricFocalTverskyLoss(nn.Module):
    def __init__(self, delta=0.7, gamma=0.75, smooth=1e-7):
        super(AsymmetricFocalTverskyLoss, self).__init__()
        '''This is the implementation for binary segmentation.
        Parameters
        ----------
        delta : float, optional
            controls weight given to false positive and false negatives, by default 0.7
        gamma : float, optional
            focal parameter controls degree of down-weighting of easy examples, by default 0.75
        smooth : float, optional
            smooithing constant to prevent division by 0 errors, by default 1
        '''
        self.delta = delta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        assert inputs.shape[1] > 1, 'The shape of model\'s output must be (b, c, h, w, ...)'
        inputs = torch.clamp(inputs, min=epsilon, max=1-epsilon)
        tp, fp, fn = get_tp_fp_fn(inputs, targets)
        dice_class = (tp+self.smooth) / (tp+self.delta*fn+(1-self.delta)*fp+self.smooth)

        back_dice = 1 - dice_class[:,0]
        fore_dice = 1 - dice_class[:,1]
        fore_dice = torch.pow(fore_dice, 1-self.gamma)

        loss = torch.stack((back_dice, fore_dice), dim=1).mean()

        # adjusts loss to account for number of classes
        # num_classes = inputs.shape[1]
        # loss = loss / num_classes
        return loss


class UnifiedFocalLoss(nn.Module):
    def __init__(self, weight=0.5, delta=0.6, gamma=0.2):
        super(UnifiedFocalLoss, self).__init__()
        '''The Unified Focal loss is a new compound loss function that unifies Dice-based and cross entropy-based loss functions into a single framework.
        Parameters
        ----------
        weight : float, optional
            represents lambda parameter and controls weight given to Asymmetric Focal Tversky loss and Asymmetric Focal loss, by default 0.5
        delta : float, optional
            controls weight given to each class, by default 0.6
        gamma : float, optional
            focal parameter controls the degree of background suppression and foreground enhancement, by default 0.2
        '''
        self.weight = weight
        self.delta = delta
        self.gamma = gamma
        self.asymmetric_focal_tversky_loss = AsymmetricFocalTverskyLoss(delta=delta, gamma=gamma)
        self.asymmetric_focal_loss = AsymmetricFocalLoss(delta=delta, gamma=gamma)

    def forward(self, inputs, targets):
      asymmetric_ftl = self.asymmetric_focal_tversky_loss(inputs, targets)
      asymmetric_fl = self.asymmetric_focal_loss(inputs, targets)
      if self.weight is not None:
        return self.weight*asymmetric_ftl + (1-self.weight)*asymmetric_fl
      else:
        return asymmetric_ftl + asymmetric_fl


# boundary Loss
class SurfaceLoss(nn.Module):
    def __init__(self):
        super(SurfaceLoss, self).__init__()

    def forward(self, inputs, dist_map):
        multipled = einsum('bcwhd, bcwhd->bcwhd', inputs[:,-1:,...], dist_map[:,-1:,...])
        loss = multipled.mean()
        return loss

import torch.nn.functional as F
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets,  reduction='none')
        pt = torch.exp(-BCE_loss)
        alpha_t = targets * self.alpha + (1- targets) * (1-self.alpha)
        F_loss = alpha_t * (1-pt)**self.gamma * BCE_loss
        #F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss