# adapted from torchvision.transforms.tranforms.py
# we are using the classes, but changing specific function, it is called class inheritance in python

import torch
import math
import random
from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
from collections.abc import Sequence, Iterable
import warnings

from torchvision.transforms import functional as F
from torchvision.transforms import RandomResizedCrop,Normalize,RandomHorizontalFlip,ToTensor
import PIL
"""_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


def _get_image_size(img):
    if F._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))

class RandomResizedCropArray(object):

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        width, height = _get_image_size(img)
        area = height * width

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, img_array):
        i, j, h, w = self.get_params(img_array[0], self.scale, self.ratio)
        return [F.resized_crop(img_array[indx], i, j, h, w, self.size, self.interpolation) for indx in range(len(img_array))]

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string"""

class ToTensorArray(ToTensor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def __call__(self, pics):
        return [F.to_tensor(pic) for pic in pics]

class RandomResizedCropArray(RandomResizedCrop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def __call__(self, img_array):
        i, j, h, w = self.get_params(PIL.Image.fromarray(img_array[0]), self.scale, self.ratio)
        return [F.resized_crop(PIL.Image.fromarray(img_array[indx]), i, j, h, w, self.size, self.interpolation) for indx in range(len(img_array))]

class RandomHorizontalFlipArray(RandomHorizontalFlip):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, imgs):
        if torch.rand(1) < self.p:
            return [F.vflip(img) for img in imgs]
        return imgs

class NormalizeArray(Normalize):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def __call__(self, tensors):
        return [F.normalize(tensor, self.mean, self.std, self.inplace) for tensor in tensors]