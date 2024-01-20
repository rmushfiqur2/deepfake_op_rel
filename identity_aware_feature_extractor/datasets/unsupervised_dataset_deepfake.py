import random
import cv2
from PIL import Image, ImageFilter
import PIL
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
from .my_functions import RandomResizedCropArray, RandomHorizontalFlipArray, ToTensorArray, NormalizeArray

class MultiCropDatasetDeepfake(datasets.VisionDataset):
    def __init__(
        self,
        dataset,
        size_crops,
        nmb_crops,
        min_scale_crops,
        max_scale_crops,
        size_dataset=-1,
        return_index=False,
        pil_blur=False,
        color_distorsion_scale=1.0,
        augment=True,
        consistent_crop_augment=False
    ):
        self.dataset = dataset
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        if size_dataset >= 0:
            self.samples = self.samples[:size_dataset]
        self.return_index = return_index

        #color_transform = [get_color_distortion(color_distorsion_scale), transforms.RandomApply([RandomGaussianBlur()], p=0.5)]
        #if pil_blur:
            #color_transform = [get_color_distortion(color_distorsion_scale), transforms.RandomApply([PILRandomGaussianBlur()], p=0.5)]
        trans = []
        mean = [0.485, 0.456, 0.406]
        std = [0.228, 0.224, 0.225]
        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            randomresizedcroparray = RandomResizedCropArray(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            if augment:
                if consistent_crop_augment:
                    trans.extend([transforms.Compose([
                        randomresizedcroparray,
                        #transforms.Compose(color_transform),
                        RandomHorizontalFlipArray(p=0.5),
                        ToTensorArray(),
                        NormalizeArray(mean=mean, std=std)
                        ])
                    ] * nmb_crops[i])
                else:
                    trans.extend([transforms.Compose([
                        randomresizedcrop,
                        #transforms.Compose(color_transform),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=mean, std=std)
                        ])
                    ] * nmb_crops[i])
            else:
                trans.extend([transforms.Compose([
                    transforms.Resize((size_crops[i],size_crops[i])),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                    ])
                ] * nmb_crops[i])
        self.trans = trans
        self.augment = augment
        self.consistent_crop_augment = consistent_crop_augment

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        # it works on a single image
        # expected output is a 4-D tensor of size (num_crops*images_per_crop,3,224,224)
        reslt = self.dataset.__getitem__(index)
        if len(reslt) < 6:
            image_raw, image_recons, feature_raw, feature_recons = reslt
        else:
            image_raw, image_recons, image_double_recons, feature_raw, feature_recons, feature_double_recons = reslt
        if self.augment:
            if self.consistent_crop_augment:
                ## need to be implement
                if len(reslt) > 4:
                    stacked_imgs = np.stack((image_raw, image_recons, image_double_recons), axis=0) # axis=0 adds new dimension at beginning
                    multi_crops_stacked = list(map(lambda trans: trans(stacked_imgs),
                                                   self.trans))  # list(len=num_crops*images_per_crop) of list(len=3) of torch.tensor(3,224,224)
                    multi_crops_stacked = [element for row in multi_crops_stacked for element in
                                           row]  # make 2D list 1D, each element is now a tensor
                    multi_crops_stacked = torch.stack((multi_crops_stacked[0], multi_crops_stacked[1], multi_crops_stacked[2]),
                                                      dim=0)  # 4D numpy torch.tensor (3,3,224,224)
                    multi_crops_raw = multi_crops_stacked[
                                      0:1]  # [0] will reduce the dimension, we don't want that, expected is 4D, 1st dimension is for num_crops
                    multi_crops_recons = multi_crops_stacked[1:2]
                    multi_crops_double = multi_crops_stacked[2:3]
                else:
                    stacked_imgs = np.stack((image_raw, image_recons), axis=0) # 4D numpy array (2,3,224,224)
                    multi_crops_stacked = list(map(lambda trans: trans(stacked_imgs), self.trans)) # list(len=num_crops*images_per_crop) of list(len=2) of torch.tensor(3,224,224)
                    multi_crops_stacked = [element for row in multi_crops_stacked for element in row] # make 2D list 1D, each element is now a tensor
                    multi_crops_stacked = torch.stack((multi_crops_stacked[0], multi_crops_stacked[1]), dim=0) # 4D numpy torch.tensor (2,3,224,224)

                    multi_crops_raw = multi_crops_stacked[0:1] # [0] will reduce the dimension, we don't want that, expected is 4D, 1st dimension is for num_crops
                    multi_crops_recons = multi_crops_stacked[1:2]
            else:
                multi_crops_raw = list(map(lambda trans: trans(PIL.Image.fromarray(image_raw)), self.trans))
                multi_crops_recons = list(map(lambda trans: trans(PIL.Image.fromarray(image_recons)), self.trans))
                if len(reslt) > 4:
                    multi_crops_double = list(map(lambda trans: trans(PIL.Image.fromarray(image_double_recons)), self.trans))
        else:
            # map(lamda trans: trans(img), self.trans) iterates over self.trans having num_crops*images_per_crop
            multi_crops_raw = list(map(lambda trans: trans(PIL.Image.fromarray(image_raw)), self.trans))
            multi_crops_recons = list(map(lambda trans: trans(PIL.Image.fromarray(image_recons)), self.trans))
            if len(reslt) > 4:
                multi_crops_double = list(map(lambda trans: trans(PIL.Image.fromarray(image_double_recons)), self.trans))
        if len(reslt) < 6:
            return multi_crops_raw, multi_crops_recons, feature_raw, feature_recons
        else:
            return multi_crops_raw, multi_crops_recons, multi_crops_double, feature_raw, feature_recons, feature_double_recons


class RandomGaussianBlur(object):
    def __call__(self, img):
        # do_it = np.random.rand() > 0.5
        # if not do_it:
        #     return img
        sigma = np.random.rand() * 1.9 + 0.1
        return Image.fromarray(cv2.GaussianBlur(np.asarray(img), (23, 23), sigma))

class PILRandomGaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.
    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """

    def __init__(self, radius_min=0.1, radius_max=2.):
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )

def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort
