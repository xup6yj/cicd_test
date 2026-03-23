# -*- coding: utf-8 -*-
import os
import nibabel
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.transform import rescale

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tF

import torchio as tio
from torchio.transforms.intensity_transform import IntensityTransform


def chwd2cdhw(tensor):
    return tensor.permute(0, 3, 1, 2)


def cdhw2chwd(tensor):
    return tensor.permute(0, 2, 3, 1)


def bchwd2bcdhw(tensor):
    return tensor.permute(0, 1, 4, 2, 3)


def bcdhw2bchwd(tensor):
    return tensor.permute(0, 1, 3, 4, 2)


def resize_image(image):
    image = chwd2cdhw(image)
    h, w = image.shape[-2], image.shape[-1]
    new_h, new_w = int(h/2), int(w/2)
    image = tF.resize(image, [new_h, new_w], interpolation=tF.InterpolationMode.BILINEAR)
    return cdhw2chwd(image)


def clipping(image):
    return torch.clip(image, 0, 1)


def affine_z(p, x, inv=False):
    R = p[:,range(1)]
    T = p[:,range(1, 3)]
    R_cos = torch.cos(R)
    R_sin = torch.sin(R)
    R = torch.eye(4, device=x.device).unsqueeze(0).repeat(x.size(0), 1, 1)
    R[:,1,1] = R_cos[:,0]
    R[:,2,2] = R_cos[:,0]
    R[:,2,1] = R_sin[:,0]
    R[:,1,2] = -R_sin[:,0]
    M = R
    M[:,range(1, 3),3] = T

    if inv:
        M = torch.linalg.inv(M)

    grid = F.affine_grid(M[:,range(3),:], x.shape, align_corners=False)
    x_affine = F.grid_sample(x, grid, align_corners=False)
    return x_affine


def flip_z(p, x):
    R = p[:,range(1)]
    T = p[:,range(1, 3)]
    R_cos = torch.cos(R)
    R_sin = torch.sin(R)
    R = torch.eye(4, device=x.device).unsqueeze(0).repeat(x.size(0), 1, 1)
    R[:,1,1] = R_cos[:,0]
    R[:,2,2] = R_cos[:,0]
    R[:,2,1] = R_sin[:,0]
    R[:,1,2] = -R_sin[:,0]
    M = R
    M[:,range(1, 3),3] = T

    M_inv = torch.linalg.inv(M)

    M_flip = torch.eye(4, device=x.device)
    M_flip[2,2] = -1
    M_flip = M_flip.unsqueeze(0).repeat(x.size(0), 1, 1)

    M = M @ M_flip @ M_inv

    grid = F.affine_grid(M[:,range(3),:], x.shape, align_corners=False)
    x_flip = F.grid_sample(x, grid, align_corners=False)
    return x_flip


def polarize(array, thres=0.5):
    array[array>=thres] = 1
    array[array<thres] = 0
    return array


def tensor2numpy(array):
    if isinstance(array, torch.Tensor):
        return array.squeeze().type(torch.float32).cpu().detach().numpy()
    return array


def imshow_tensor(tensor, title=None, save_dir=None, save_fn=None):
    image = tensor.squeeze().cpu().detach().numpy()
    plt.imshow(image, cmap='gray')
    if title is not None:
        plt.title(title)
    if save_dir is None:
        plt.show()
    else:
        plt.savefig(os.path.join(save_dir, save_fn), dpi=150)
    plt.clf()


def plot_slice(image, mask, output, prob, save_dir, save_fn):
    image = tensor2numpy(image)
    if mask is None and output is None:
        fig, ax = plt.subplots(ncols=1, sharey=True)
        ax.imshow(image, cmap='gray', vmin=0, vmax=1)
        ax.set_title('CT')
    else:
        output = tensor2numpy(output)
        overlay_pred = np.dstack(((1-output)*image+output, image, image))
        extra_col = int(prob is not None)
        if mask is None:
            # for testing (no ground truth)
            fig, axes = plt.subplots(nrows=2, ncols=2+extra_col, sharex=True, sharey=True)
            axes = axes.flatten()

            axes[0].imshow(image, cmap='gray', vmin=0, vmax=1)
            axes[0].set_title('CT')

            axes[1].imshow(output, cmap='gray', vmin=0, vmax=1)
            axes[1].set_title('Pred')

            axes[3+extra_col].imshow(overlay_pred, vmin=0, vmax=1)
            axes[3+extra_col].set_title('Pred (overlay)')

            if prob is not None:
                axes[2].imshow(prob, cmap='gist_heat', vmin=0, vmax=1)
                axes[2].set_title('')

            for i in range(len(axes)):
                axes[i].axis('off')
        else:
            # for evaluation (comparing with the ground truth)
            mask = tensor2numpy(mask)
            overlay_gt = np.dstack(((1-mask)*image+mask, image, image))
            fig, axes = plt.subplots(nrows=2, ncols=3+extra_col, sharex=True, sharey=True)
            axes = axes.flatten()

            axes[0].imshow(image, cmap='gray', vmin=0, vmax=1)
            axes[0].set_title('CT')

            axes[1].imshow(mask, cmap='gray', vmin=0, vmax=1)
            axes[1].set_title('GT')

            axes[4+extra_col].imshow(overlay_gt, vmin=0, vmax=1)
            axes[4+extra_col].set_title('GT (overlay)')

            axes[2].imshow(output, cmap='gray', vmin=0, vmax=1)
            axes[2].set_title('Pred')

            axes[5+extra_col].imshow(overlay_pred, vmin=0, vmax=1)
            axes[5+extra_col].set_title('Pred (overlay)')

            if prob is not None:
                axes[3].imshow(prob, cmap='gist_heat', vmin=0, vmax=1)
                axes[3].set_title('')

            for i in range(len(axes)):
                axes[i].axis('off')

    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, save_fn), dpi=150)
    plt.clf()
    plt.close()


def plot_cr(ori_image, corrupt, restore, save_dir, save_fn):
    ori_image = tensor2numpy(ori_image)
    corrupt = tensor2numpy(corrupt)
    restore = tensor2numpy(restore)
    fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True)
    axes = axes.flatten()

    axes[0].imshow(ori_image, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('origin')

    axes[1].imshow(corrupt, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('corrupt')

    axes[2].imshow(restore, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('restore')

    for i in range(len(axes)):
        axes[i].axis('off')

    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, save_fn), dpi=150)
    plt.clf()
    plt.close()


def plot_adn(A, P, Q, X, XP, XQ, save_dir, save_fn, degree=90):
    A = tensor2numpy(A)
    P = tensor2numpy(P)
    Q = tensor2numpy(Q)
    X = tensor2numpy(X)
    XP = tensor2numpy(XP)
    XQ = tensor2numpy(XQ)

    A = ndimage.rotate(tensor2numpy(A), degree)
    P = ndimage.rotate(tensor2numpy(P), degree)
    Q = ndimage.rotate(tensor2numpy(Q), degree)
    X = ndimage.rotate(tensor2numpy(X), degree)
    XP = ndimage.rotate(tensor2numpy(XP), degree)
    XQ = ndimage.rotate(tensor2numpy(XQ), degree)

    fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
    axes = axes.flatten()

    axes[0].imshow(A, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('A')

    axes[1].imshow(P, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('P')

    axes[2].imshow(Q, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('Q')

    axes[3].imshow(X, cmap='gray', vmin=0, vmax=1)
    axes[3].set_title('X')

    axes[4].imshow(XP, cmap='gray', vmin=0, vmax=1)
    axes[4].set_title('X_flip')

    axes[5].imshow(XQ, cmap='gray', vmin=0, vmax=1)
    axes[5].set_title('X+Q')

    for i in range(len(axes)):
        axes[i].axis('off')

    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, save_fn), dpi=150)
    plt.clf()
    plt.close()


def plot_features(x_features, x_flip_features, save_dir, save_fn, nrows=2, ncols=3, degree=270):
    channels = x_features.shape[0]
    X = tensor2numpy(x_features)
    X_flip = tensor2numpy(x_flip_features)

    X = ndimage.rotate(np.transpose(X, (2, 1, 0)), degree)
    X = np.transpose(X, (2, 1, 0))
    X_flip = ndimage.rotate(np.transpose(X_flip, (2, 1, 0)), degree)
    X_flip = np.transpose(X_flip, (2, 1, 0))

    iters = channels // ncols
    for i in range(iters):
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)
        axes = axes.flatten()

        for col in range(ncols):
            axes[col].imshow(X[col+i*ncols], cmap='gray', vmin=0, vmax=1)
            axes[col].set_title(f'{col+i*ncols+1}')

        for col in range(ncols):
            axes[col+ncols].imshow(X_flip[col+i*ncols], cmap='gray', vmin=0, vmax=1)
            axes[col+ncols].set_title(f'{col+i*ncols+1}')
        
        for j in range(len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'ch_{i*ncols+1}-{i*ncols+ncols}.jpg'), dpi=150)
        plt.clf()
        plt.close()


def rescale_features(features):
    # input size: (B, C, H, W, D)
    b, c, h, w, d = features.shape
    if h == 256:
        return features
    features = tensor2numpy(features)
    scale = 256 // h

    re_features = np.zeros((b, c, h*scale, w*scale, d))
    for batch in range(b):
        for channel in range(c):
            feat = features[batch, channel, ...]
            feat = rescale(feat, (scale, scale, 1), order=1, preserve_range=True)
            re_features[batch, channel, ...] = feat
    
    return torch.from_numpy(re_features)


def save_nii(ori_nii_image_path, pred, save_dir, save_fn):
    ori = nibabel.load(ori_nii_image_path)
    seg_result = nibabel.Nifti1Image(pred.detach().numpy(), ori.affine, ori.header)
    nibabel.save(seg_result, os.path.join(save_dir, save_fn))


def generate_model_string(args):
    if args.model in ['R2UNet', 'RUNet']:
        # R2UNet, RUNet
        model = '{}(in_ch={}, out_ch={}, init_ch={}, num_rcnn={}, t={}, deconv={}, cgm={})'.format(
                    args.model, args.in_channel, args.out_channel, args.init_ch, args.num_rcnn, args.t, args.deconv, args.cgm)
    else:
        # UNetPP, UNet3P, APUNet, ResUNet, AttentionUNet, UNet, UNet3P
        model = '{}(in_ch={}, out_ch={}, init_ch={}, deconv={}, cgm={})'.format(
                    args.model, args.in_channel, args.out_channel, args.init_ch, args.deconv, args.cgm)
    return model


def custom_filter(image, D0=15, D1=30):
    h, w, d = image.shape
    image_ifft_3d = np.zeros(image.shape)
    for z in range(d):
        slice = image[..., z]
        f = np.fft.fft2(slice)
        fshift = np.fft.fftshift(f)
        # image_fft = np.log(np.abs(fshift))

        H1 = np.zeros((h, w), dtype=np.float32)
        for u in range(h):
            for v in range(w):
                D = np.sqrt((u-h/2)**2 + (v-w/2)**2)
                H1[u, v] = np.exp(-D**2/(2*D0*D0))

        H2 = np.zeros((h, w), dtype=np.float32)
        for u in range(h):
            for v in range(w):
                D = np.sqrt((u-h/2)**2 + (v-w/2)**2)
                H2[u, v] = np.exp(-D**2/(2*D1*D1))

        H = H2 - H1

        ishift = np.fft.ifftshift(fshift*(1-H))
        image_ifft = np.fft.ifft2(ishift)
        image_ifft = np.abs(image_ifft)
        image_ifft_3d[..., z] = image_ifft

    return image_ifft_3d


class CopyAffine(IntensityTransform):
    def _init_(self, **kwargs):
        super(CopyAffine, self). _init_(**kwargs)

    def apply_transform(self, subject):
        transform = tio.CopyAffine('image')
        subject = transform(subject)
        return subject