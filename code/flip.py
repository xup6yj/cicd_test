# -*- coding: utf-8 -*-
import os
from glob import glob
import nibabel as nib
from tqdm import tqdm

import torch

from utils import flip_z
from models import RoughAffineZ

device = 'cuda:0'

def save_nii(ori_nii, image, save_dir, save_fn):
    ori = nib.load(ori_nii)
    image = nib.Nifti1Image(image.detach().cpu().numpy(), ori.affine, ori.header)
    nib.save(image, os.path.join(save_dir, save_fn))

if __name__ == '__main__':
    affine_model = RoughAffineZ().to(device)
    affine_model_path = 'affine_z_lvl4_ssim.cpt'
    affine_model.load_state_dict(torch.load(affine_model_path, map_location=device)['model'])
    affine_model.eval()

    root = '/media/adchentc/Andy/UI/cgmh_web_DWI/static/patient_data'
    fpath = sorted(glob(os.path.join(root, '*')))

    for subject in tqdm(fpath):
        dest = os.path.join(root, subject)
        nii = os.path.join(dest, 'CT_ss.nii.gz')

        image = nib.load(nii).get_fdata()
        h, w, d = image.shape
        image = torch.Tensor(image).to(device).unsqueeze(dim=0).unsqueeze(dim=1)
        with torch.no_grad():
            p = affine_model(image)
        image = flip_z(p, image).reshape(h, w, d)
        save_nii(nii, image, dest, 'CT_flip.nii.gz')