# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
from os.path import join
from scipy.ndimage import distance_transform_edt

import torch
import torchvision.transforms.functional as tF

import torchio as tio
from torchio.data import SubjectsDataset
from torchio.transforms.spatial_transform import SpatialTransform
from torchio.transforms.intensity_transform import IntensityTransform

from utils import chwd2cdhw, cdhw2chwd, flip_z, tensor2numpy
from utils import CopyAffine
from models import RoughAffineZ


def get_conf_map(mask):
    mask = mask.type(torch.float32)
    k = (3, 3, 1)
    p = (1, 1, 0)
    min_pool = torch.nn.functional.max_pool3d(-mask, k, 1, p) * -1
    max_min_pool = torch.nn.functional.max_pool3d(min_pool, k, 1, p)
    contour = torch.relu(max_min_pool-min_pool)
    non_contour = 1 - contour
    conf_map = np.zeros(contour.shape)
    for z in range(conf_map.shape[-1]):
        if contour[...,z].any():
            conf_map[...,z] = distance_transform_edt(non_contour[...,z])
    conf_map = conf_map / conf_map.max()
    conf_map = (1-conf_map) ** 100
    conf_map = torch.Tensor(conf_map)
    conf_map = (conf_map+min_pool).clip(0, 1)
    return conf_map


def noisy_label_3d(label, noise_sigma, kernel_size, soft_boundary):
    if soft_boundary:
        conf_map = get_conf_map(label)
    label = chwd2cdhw(label).type(torch.float32)
    label_smooth = tF.gaussian_blur(label, kernel_size, [noise_sigma, noise_sigma])
    p_map = 0.5 - torch.abs(label_smooth-0.5)
    noise = torch.rand(size=label.shape)
    flip = torch.where(noise<p_map, 1, 0)
    noisy_label = torch.logical_xor(flip, label).type(torch.float32)
    noisy_label = cdhw2chwd(noisy_label)
    if soft_boundary:
        noisy_label = noisy_label * conf_map
    return noisy_label


class NoisyLabel(IntensityTransform):
    def __init__(self, noise_mm=5, kernel_size=None, soft_boundary=False, **kwargs):
        super(NoisyLabel, self).__init__(**kwargs)
        self.noise_mm = noise_mm
        self.kernel_size = kernel_size
        self.soft_boundary = soft_boundary

    def apply_transform(self, subject):
        label = subject['label'][tio.DATA]
        assert label.ndim==4 and label.shape[0]==1
        spacing = subject['label'].spacing
        noise_voxel = self.noise_mm / spacing[0]
        if self.kernel_size is None:
            kernel_size = int(noise_voxel)//2*2 + 1
        else:
            kernel_size = self.kernel_size
        new_data = noisy_label_3d(
            label=label,
            noise_sigma=noise_voxel,
            kernel_size=kernel_size,
            soft_boundary=self.soft_boundary)
        subject['label'].set_data(new_data)
        return subject


class DistanceMap(IntensityTransform):
    def __init__(self, mode='3d', focus_boundary=False, **kwargs):
        super(DistanceMap, self).__init__(**kwargs)
        self.mode = mode
        self.focus_boundary = focus_boundary

    def apply_transform(self, subject):
        roi = subject['label'][tio.DATA]
        dist_map = np.zeros(roi.shape)
        roi = roi.numpy()
        normal = 1 - roi
        if self.mode == '3d':
            if roi.any():
                dist_map = distance_transform_edt(normal)*normal - (distance_transform_edt(roi)-1)*roi
        elif self.mode == '2d':
            for z in range(roi.shape[-1]):
                if roi[...,z].any():
                    dist_map[...,z] = distance_transform_edt(normal[...,z])*normal[...,z] - (distance_transform_edt(roi[...,z])-1)*roi[...,z]
        else:
            raise ValueError('Unknown mode')

        if self.focus_boundary:
            roi_min, roi_max = -dist_map.min(), dist_map.max()
            dist_map = (-1-dist_map/roi_min)*roi + (1-dist_map/roi_max)*normal
        subject['dist_map'].set_data(dist_map)
        return subject


class SampleProbabilityMap(IntensityTransform):
    def __init__(self, icv_weight=0, **kwargs):
        super(SampleProbabilityMap, self).__init__(**kwargs)
        self.icv_weight = icv_weight

    def apply_transform(self, subject):
        has_roi = (subject['label'][tio.DATA].sum(dim=[1, 2])>0).type(torch.float32)
        has_brain = (subject['icv'][tio.DATA].sum(dim=[1, 2])>0).type(torch.float32)
        prob_map = subject['icv'][tio.DATA] * (self.icv_weight*has_brain+(1-self.icv_weight)*has_roi)
        subject['prob_map'].set_data(prob_map)
        return subject


class RandomAffine(SpatialTransform):
    def __init__(self, degrees=0, translate=None, scale=None, **kwargs):
        super(RandomAffine, self).__init__(**kwargs)
        self.degrees = degrees
        self.translate = 0 if translate is None else translate
        self.scale = 0 if scale is None else scale

    def apply_transform(self, subject):
        angle = np.random.randint(-self.degrees, self.degrees)
        translate = np.random.rand(2) * self.translate * 256
        scale = 1 - (np.random.rand()*2-1) * self.scale
        for image in self.get_images(subject):
            new_data = tF.affine(
                            chwd2cdhw(image.data),
                            angle=angle, translate=translate.tolist(),
                            scale=scale, shear=[0, 0])
            new_data = cdhw2chwd(new_data)
            image.set_data(new_data)
        return subject


class Flip(IntensityTransform):
    def __init__(self, **kwargs):
        super(Flip, self).__init__(**kwargs)
        self.affine_model = RoughAffineZ()
        affine_model_path = 'affine_z_lvl4.cpt'
        self.affine_model.load_state_dict(torch.load(affine_model_path, map_location='cpu')['model'])
        self.affine_model.eval()

    def apply_transform(self, subject):
        for image in self.get_images(subject):
            image_extend = image.data.unsqueeze(dim=0)
            with torch.no_grad():
                p = self.affine_model(image_extend)
            image_flip = flip_z(p, image_extend)
            image.set_data(image_flip.squeeze(dim=0))
        return subject


# dataset for in-house / AISD
class CTDataset(SubjectsDataset):
    '''
    NCCT: `data_dir`/images/.*nii.gz
    flipped NCCT: `data_dir`/images_flip/.*nii.gz
    ROI: `data_dir`/masks/.*nii.gz
    ICV: `data_dir`/ICV_masks/.*nii.gz
    '''
    def __init__(
            self, data_dir, mode='train',
            flip_p=0.5, trans_range=0.1, resize_range=0.1, rotate_range=20, noise_mm=10,
            no_noisy=False, flip_image=True, curriculum=True, **kwargs):
        flip_p = 0 if flip_p is None else flip_p

        image_dir = join(data_dir, 'images')
        image_paths = sorted(glob.glob(join(image_dir, '*.nii.gz')))
        if flip_image:
            image_flip_paths = [path.replace('images', 'images_flip') for path in image_paths]
        else:
            image_flip_paths = image_paths
        roi_paths = [path.replace('images', 'masks') for path in image_paths]
        icv_paths = [path.replace('images', 'ICV_masks') for path in image_paths]

        image_num = len(image_paths)
        file_names = [os.path.split(x)[1] for x in image_paths]
        subj_names = [os.path.splitext(os.path.splitext(x)[0])[0] for x in file_names]

        if mode == 'train':
            subjects = [
                tio.Subject(
                    image=tio.ScalarImage(image_paths[i]),
                    image_flip=tio.ScalarImage(image_flip_paths[i]),
                    label=tio.ScalarImage(roi_paths[i]),
                    dist_map=tio.ScalarImage(roi_paths[i]),
                    prob_map=tio.ScalarImage(icv_paths[i]),
                    icv=tio.ScalarImage(icv_paths[i]),
                    image_path=image_paths[i],
                    name=subj_names[i])
                for i in range(image_num)
            ]
        elif mode == 'val':
            subjects = [
                tio.Subject(
                    image=tio.ScalarImage(image_paths[i]),
                    image_flip=tio.ScalarImage(image_flip_paths[i]),
                    label=tio.ScalarImage(roi_paths[i]),
                    dist_map=tio.ScalarImage(roi_paths[i]),
                    image_path=image_paths[i],
                    name=subj_names[i])
                for i in range(image_num)
            ]
        else:
            subjects = [
                tio.Subject(
                    image=tio.ScalarImage(image_paths[i]),
                    image_flip=tio.ScalarImage(image_flip_paths[i]),
                    image_path=image_paths[i],
                    name=subj_names[i])
                for i in range(image_num)
            ]

        transform = []
        # if flip_image:
        #     transform.append(Flip(include=['image_flip']))

        if mode == 'train':
            transform += [
                tio.transforms.RandomFlip(axes='LR', flip_probability=flip_p),
                RandomAffine(rotate_range, trans_range, resize_range),
            ]
            if not no_noisy:
                transform.append(NoisyLabel(noise_mm=noise_mm, include=['label']))
            if curriculum:
                transform.append(tio.transforms.Pad((64, 64, 4)))
                transform.append(SampleProbabilityMap(icv_weight=0, include=['prob_map']))
        if mode=='train' or mode=='val':
            transform.append(DistanceMap(include=['dist_map']))
        transform = tio.transforms.Compose(transform)
        super(CTDataset, self).__init__(subjects=subjects, transform=transform, **kwargs)


# dataset for affine net training
class AffineDataset(SubjectsDataset):
    '''
    NCCT: `data_dir`/images/.*nii.gz
    ICV: `data_dir`/ICV_masks/.*nii.gz
    '''
    def __init__(
            self, data_dir, mode='train',
            flip_p=0.5, trans_range=0.1, resize_range=0.1, rotate_range=20,
            **kwargs):
        flip_p = 0 if flip_p is None else flip_p

        image_dir = join(data_dir, 'images')

        image_paths = sorted(glob.glob(join(image_dir, '*.nii.gz')))
        icv_paths = [path.replace('images', 'ICV_masks') for path in image_paths]

        image_num = len(image_paths)
        file_names = [os.path.split(x)[1] for x in image_paths]
        subj_names = [os.path.splitext(os.path.splitext(x)[0])[0] for x in file_names]

        subjects = [
            tio.Subject(
                image=tio.ScalarImage(image_paths[i]),
                icv=tio.ScalarImage(icv_paths[i]),
                image_path=image_paths[i],
                name=subj_names[i])
            for i in range(image_num)
        ]

        transform = []
        if mode == 'train':
            transform += [
                tio.transforms.RandomFlip(axes='LR', flip_probability=flip_p),
                RandomAffine(rotate_range, trans_range, resize_range),
            ]
        transform = tio.transforms.Compose(transform)
        super(AffineDataset, self).__init__(subjects=subjects, transform=transform, **kwargs)


# dataset for context restoration pre-training
class CRDataset(SubjectsDataset):
    '''
    NCCT: `data_dir`/images/.*nii.gz
    '''
    def __init__(
            self, data_dir,
            flip_p=0.5, trans_range=0.1, resize_range=0.1, rotate_range=20,
            **kwargs):
        flip_p = 0 if flip_p is None else flip_p

        image_dir = join(data_dir, 'images')

        image_paths = sorted(glob.glob(join(image_dir, '*.nii.gz')))
        image_num = len(image_paths)

        file_names = [os.path.split(x)[1] for x in image_paths]
        subj_names = [os.path.splitext(os.path.splitext(x)[0])[0] for x in file_names]

        subjects = [
            tio.Subject(
                image=tio.ScalarImage(image_paths[i]),
                image_path=image_paths[i],
                name=subj_names[i])
            for i in range(image_num)
        ]

        transform = [
            tio.transforms.RandomFlip(axes='LR', flip_probability=flip_p),
            RandomAffine(rotate_range, trans_range, resize_range),
        ]
        transform = tio.transforms.Compose(transform)
        super(CRDataset, self).__init__(subjects=subjects, transform=transform, **kwargs)

# AISD
class AISDDataset(SubjectsDataset):
    '''
    NCCT: `data_dir`/images/.*nii.gz
    flipped NCCT: `data_dir`/images_flip/.*nii.gz
    ROI: `data_dir`/masks/.*nii.gz
    ICV: `data_dir`/ICV_masks/.*nii.gz

    GM: `data_dir`/GM_masks/.*nii.gz
    flipped GM: `data_dir`/GM_masks_flip/.*nii.gz
    WM: `data_dir`/WM_masks/.*nii.gz
    flipped WM: `data_dir`/WM_masks_flip/.*nii.gz
    CSF: `data_dir`/CSF_masks/.*nii.gz
    flipped CSF: `data_dir`/CSF_masks_flip/.*nii.gz

    DWI: `data_dir`/DWI/.*nii.gz
    DWI ROI: `data_dir`/DWI_masks/.*nii.gz 
    '''
    def __init__(
            self, data_dir, mode='train',
            flip_p=0.5, trans_range=0.1, resize_range=0.1, rotate_range=20, noise_mm=10,
            no_noisy=False, flip_image=True, curriculum=True, **kwargs):
        flip_p = 0 if flip_p is None else flip_p

        image_dir = join(data_dir, 'images')
        image_paths = sorted(glob.glob(join(image_dir, '*.nii.gz')))
        if flip_image:
            image_flip_paths = [path.replace('images', 'images_flip') for path in image_paths]
        else:
            image_flip_paths = image_paths
        roi_paths = [path.replace('images', 'masks') for path in image_paths]
        icv_paths = [path.replace('images', 'ICV_masks') for path in image_paths]
        csf_paths = [path.replace('images', 'CSF_masks') for path in image_paths]
        image_num = len(image_paths)
        file_names = [os.path.split(x)[1] for x in image_paths]
        subj_names = [os.path.splitext(os.path.splitext(x)[0])[0] for x in file_names]

        if mode == 'train':
            subjects = [
                tio.Subject(
                    image=tio.ScalarImage(image_paths[i]),
                    image_flip=tio.ScalarImage(image_flip_paths[i]),
                    label=tio.ScalarImage(roi_paths[i]),
                    dist_map=tio.ScalarImage(roi_paths[i]),
                    prob_map=tio.ScalarImage(icv_paths[i]),
                    icv=tio.ScalarImage(icv_paths[i]),

                    csf=tio.ScalarImage(csf_paths[i]),

                    image_path=image_paths[i],
                    name=subj_names[i])
                for i in range(image_num)
            ]
        elif mode == 'val':
            subjects = [
                tio.Subject(
                    image=tio.ScalarImage(image_paths[i]),
                    image_flip=tio.ScalarImage(image_flip_paths[i]),
                    label=tio.ScalarImage(roi_paths[i]),
                    dist_map=tio.ScalarImage(roi_paths[i]),
                    icv=tio.ScalarImage(icv_paths[i]),

                    csf=tio.ScalarImage(csf_paths[i]),

                    image_path=image_paths[i],
                    name=subj_names[i])
                for i in range(image_num)
            ]
        else:
            subjects = [
                tio.Subject(
                    image=tio.ScalarImage(image_paths[i]),
                    image_flip=tio.ScalarImage(image_flip_paths[i]),
                    image_path=image_paths[i],
                    name=subj_names[i])
                for i in range(image_num)
            ]

        transform = []
        # if flip_image:
        #     transform.append(Flip(include=['image_flip']))

        if mode == 'train':
            transform += [
                tio.transforms.RandomFlip(axes='LR', flip_probability=flip_p),
                RandomAffine(rotate_range, trans_range, resize_range),
            ]
            if not no_noisy:
                transform.append(NoisyLabel(noise_mm=noise_mm, include=['label']))
            if curriculum:
                transform.append(tio.transforms.Pad((64, 64, 4)))
                transform.append(SampleProbabilityMap(icv_weight=0, include=['prob_map']))
        if mode=='train' or mode=='val':
            transform.append(DistanceMap(include=['dist_map']))
        transform = tio.transforms.Compose(transform)
        super(AISDDataset, self).__init__(subjects=subjects, transform=transform, **kwargs)


# dataset for 52 testing data
class AISTestDataset(SubjectsDataset):
    '''
    NCCT: `data_dir`/images/.*nii.gz
    flipped NCCT: `data_dir`/images_flip/.*nii.gz
    ROI: `data_dir`/masks/.*nii.gz
    '''
    def __init__(self, data_dir, **kwargs):
        image_paths = sorted(glob.glob(join(data_dir, 'images/*.nii.gz')))
        image_num = len(image_paths)

        subj_names = [x.split('/')[-1].split('.')[0] for x in image_paths]

        image_flip_paths = [x.replace('images', 'images_flip') for x in image_paths]
        roi_paths = [x.replace('images', 'masks') for x in image_paths]

        subjects = [
            tio.Subject(
                image=tio.ScalarImage(image_paths[i]),
                image_flip=tio.ScalarImage(image_flip_paths[i]),
                label=tio.ScalarImage(roi_paths[i]),
                image_path=image_paths[i],
                name=subj_names[i])
            for i in range(image_num)
        ]

        transform = []
        # transform = [Flip(include=['image_flip'])]
        transform = tio.transforms.Compose(transform)
        super(AISTestDataset, self).__init__(subjects=subjects, transform=transform, **kwargs)


# dataset for new in-house
class NCCT111Dataset(SubjectsDataset):
    '''
    NCCT: `data_dir`/images/.*nii.gz
    flipped NCCT: `data_dir`/images_flip/.*nii.gz
    ROI: `data_dir`/masks/.*nii.gz
    ICV: `data_dir`/ICV_masks/.*nii.gz

    GM: `data_dir`/GM_masks/.*nii.gz
    flipped GM: `data_dir`/GM_masks_flip/.*nii.gz
    WM: `data_dir`/WM_masks/.*nii.gz
    flipped WM: `data_dir`/WM_masks_flip/.*nii.gz
    CSF: `data_dir`/CSF_masks/.*nii.gz
    flipped CSF: `data_dir`/CSF_masks_flip/.*nii.gz

    DWI: `data_dir`/DWI/.*nii.gz
    DWI ROI: `data_dir`/DWI_masks/.*nii.gz 
    '''
    def __init__(
            self, data_dir, mode='train',
            flip_p=0.5, trans_range=0.1, resize_range=0.1, rotate_range=20, noise_mm=10,
            no_noisy=False, flip_image=True, curriculum=True, **kwargs):
        flip_p = 0 if flip_p is None else flip_p

        image_dir = join(data_dir, 'images')
        image_paths = sorted(glob.glob(join(image_dir, '*.nii.gz')))
        if flip_image:
            image_flip_paths = [path.replace('images', 'images_flip') for path in image_paths]
        else:
            image_flip_paths = image_paths
        roi_paths = [path.replace('images', 'masks') for path in image_paths]
        icv_paths = [path.replace('images', 'ICV_masks') for path in image_paths]

        # gm_paths = [path.replace('images', 'GM_masks') for path in image_paths]
        # gm_flip_paths = [path.replace('images', 'GM_masks_flip') for path in image_paths]
        # wm_paths = [path.replace('images', 'WM_masks') for path in image_paths]
        # wm_flip_paths = [path.replace('images', 'WM_masks_flip') for path in image_paths]
        csf_paths = [path.replace('images', 'CSF_masks') for path in image_paths]

        # csf_flip_paths = [path.replace('images', 'CSF_masks_flip') for path in image_paths]
        # pcsf_paths = [path.replace('images', 'pcsf') for path in image_paths]
        # dwi_paths = [path.replace('images', 'DWI') for path in image_paths]
        # dwi_roi_paths = [path.replace('images', 'DWI_masks') for path in image_paths]
        # tmp_paths = [path.replace('images', 'tmp') for path in image_paths]

        image_num = len(image_paths)
        file_names = [os.path.split(x)[1] for x in image_paths]
        subj_names = [os.path.splitext(os.path.splitext(x)[0])[0] for x in file_names]

        if mode == 'train':
            subjects = [
                tio.Subject(
                    image=tio.ScalarImage(image_paths[i]),
                    image_flip=tio.ScalarImage(image_flip_paths[i]),
                    label=tio.ScalarImage(roi_paths[i]),
                    dist_map=tio.ScalarImage(roi_paths[i]),
                    prob_map=tio.ScalarImage(icv_paths[i]),
                    icv=tio.ScalarImage(icv_paths[i]),
                    csf=tio.ScalarImage(csf_paths[i]),

                    # gm=tio.ScalarImage(gm_paths[i]),
                    # gm_flip=tio.ScalarImage(gm_flip_paths[i]),
                    # wm=tio.ScalarImage(wm_paths[i]),
                    # wm_flip=tio.ScalarImage(wm_flip_paths[i]),
                    
                    # csf_flip=tio.ScalarImage(csf_flip_paths[i]),
                    # pcsf=tio.ScalarImage(pcsf_paths[i]),

                    # dwi=tio.ScalarImage(dwi_paths[i]),
                    # dwi_roi=tio.ScalarImage(dwi_roi_paths[i]),

                    # tmp=tio.ScalarImage(tmp_paths[i]),

                    image_path=image_paths[i],
                    name=subj_names[i])
                for i in range(image_num)
            ]
        elif mode == 'val':
            subjects = [
                tio.Subject(
                    image=tio.ScalarImage(image_paths[i]),
                    image_flip=tio.ScalarImage(image_flip_paths[i]),
                    label=tio.ScalarImage(roi_paths[i]),
                    dist_map=tio.ScalarImage(roi_paths[i]),
                    icv=tio.ScalarImage(icv_paths[i]),

                    # gm=tio.ScalarImage(gm_paths[i]),
                    # gm_flip=tio.ScalarImage(gm_flip_paths[i]),
                    # wm=tio.ScalarImage(wm_paths[i]),
                    # wm_flip=tio.ScalarImage(wm_flip_paths[i]),
                    csf=tio.ScalarImage(csf_paths[i]),
                    # csf_flip=tio.ScalarImage(csf_flip_paths[i]),
                    # pcsf=tio.ScalarImage(pcsf_paths[i]),

                    # tmp=tio.ScalarImage(tmp_paths[i]),

                    image_path=image_paths[i],
                    name=subj_names[i])
                for i in range(image_num)
            ]
        else:
            subjects = [
                tio.Subject(
                    image=tio.ScalarImage(image_paths[i]),
                    image_flip=tio.ScalarImage(image_flip_paths[i]),
                    image_path=image_paths[i],
                    name=subj_names[i])
                for i in range(image_num)
            ]

        transform = []
        # if flip_image:
        #     transform.append(Flip(include=['image_flip']))

        if mode == 'train':
            transform += [
                tio.transforms.RandomFlip(axes='LR', flip_probability=flip_p),
                RandomAffine(rotate_range, trans_range, resize_range),
            ]
            if not no_noisy:
                transform.append(NoisyLabel(noise_mm=noise_mm, include=['label']))
            if curriculum:
                # transform.append(tio.transforms.Pad((64, 64, 4)))
                transform.append(SampleProbabilityMap(icv_weight=0, include=['prob_map']))
        if mode=='train' or mode=='val':
            transform.append(DistanceMap(include=['dist_map']))
        transform = tio.transforms.Compose(transform)
        super(NCCT111Dataset, self).__init__(subjects=subjects, transform=transform, **kwargs)


# dataset for new in-house
class PreStrokeDataset(SubjectsDataset):
    '''
    NCCT: `data_dir`/images/.*nii.gz
    flipped NCCT: `data_dir`/images_flip/.*nii.gz
    ROI: `data_dir`/masks/.*nii.gz
    ICV: `data_dir`/ICV_masks/.*nii.gz

    GM: `data_dir`/GM_masks/.*nii.gz
    flipped GM: `data_dir`/GM_masks_flip/.*nii.gz
    WM: `data_dir`/WM_masks/.*nii.gz
    flipped WM: `data_dir`/WM_masks_flip/.*nii.gz
    CSF: `data_dir`/CSF_masks/.*nii.gz
    flipped CSF: `data_dir`/CSF_masks_flip/.*nii.gz

    DWI: `data_dir`/DWI/.*nii.gz
    DWI ROI: `data_dir`/DWI_masks/.*nii.gz 
    '''
    def __init__(
            self, data_dir, mode='train',
            flip_p=0.5, trans_range=0.1, resize_range=0.1, rotate_range=20, noise_mm=10,
            no_noisy=False, flip_image=True, curriculum=True, **kwargs):
        flip_p = 0 if flip_p is None else flip_p

        image_dir = join(data_dir, 'images')
        image_paths = sorted(glob.glob(join(image_dir, '*.nii.gz')))
        if flip_image:
            image_flip_paths = [path.replace('images', 'images_flip') for path in image_paths]
        else:
            image_flip_paths = image_paths
        roi_paths = [path.replace('images', 'masks') for path in image_paths]
        icv_paths = [path.replace('images', 'ICV_masks') for path in image_paths]
        csf_paths = [path.replace('images', 'p_masks') for path in image_paths]

        image_num = len(image_paths)
        file_names = [os.path.split(x)[1] for x in image_paths]
        subj_names = [os.path.splitext(os.path.splitext(x)[0])[0] for x in file_names]

        if mode == 'train':
            subjects = [
                tio.Subject(
                    image=tio.ScalarImage(image_paths[i]),
                    image_flip=tio.ScalarImage(image_flip_paths[i]),
                    label=tio.ScalarImage(roi_paths[i]),
                    dist_map=tio.ScalarImage(roi_paths[i]),
                    prob_map=tio.ScalarImage(icv_paths[i]),
                    icv=tio.ScalarImage(icv_paths[i]),
                    csf=tio.ScalarImage(csf_paths[i]),
                    image_path=image_paths[i],
                    name=subj_names[i])
                for i in range(image_num)
            ]
        elif mode == 'val':
            subjects = [
                tio.Subject(
                    image=tio.ScalarImage(image_paths[i]),
                    image_flip=tio.ScalarImage(image_flip_paths[i]),
                    label=tio.ScalarImage(roi_paths[i]),
                    dist_map=tio.ScalarImage(roi_paths[i]),
                    icv=tio.ScalarImage(icv_paths[i]),
                    csf=tio.ScalarImage(csf_paths[i]),
                    image_path=image_paths[i],
                    name=subj_names[i])
                for i in range(image_num)
            ]
        else:
            subjects = [
                tio.Subject(
                    image=tio.ScalarImage(image_paths[i]),
                    image_flip=tio.ScalarImage(image_flip_paths[i]),
                    image_path=image_paths[i],
                    name=subj_names[i])
                for i in range(image_num)
            ]

        transform = []
        # if flip_image:
        #     transform.append(Flip(include=['image_flip']))

        if mode == 'train':
            transform += [
                tio.transforms.RandomFlip(axes='LR', flip_probability=flip_p),
                RandomAffine(rotate_range, trans_range, resize_range),
            ]
            if not no_noisy:
                transform.append(NoisyLabel(noise_mm=noise_mm, include=['label']))
            if curriculum:
                # transform.append(tio.transforms.Pad((64, 64, 4)))
                transform.append(SampleProbabilityMap(icv_weight=0, include=['prob_map']))
        if mode=='train' or mode=='val':
            transform.append(DistanceMap(include=['dist_map']))
        transform = tio.transforms.Compose(transform)
        super(PreStrokeDataset, self).__init__(subjects=subjects, transform=transform, **kwargs)