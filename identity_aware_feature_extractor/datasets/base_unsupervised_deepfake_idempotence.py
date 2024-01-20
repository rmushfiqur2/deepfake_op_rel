
import numpy as np
import io
import h5py
from PIL import Image
from torch.utils.data import Dataset
from os.path import isfile, isdir, split
import os
import glob
import pickle

class DatasetBaseUnsupervised_deepfake_idempotence(Dataset):
    def __init__(self, root, transform=None):
        assert os.path.exists(root), '{} does not exist!'.format(root)
        self.root = root

        # Create a wildcard pattern to match all files in the folder
        pattern = os.path.join(root, '*.pickle')

        # Use glob to retrieve a list of matching file paths
        self.file_list = glob.glob(pattern)
        self.length = len(self.file_list)
        self.transform = transform

        print('Base unsupervised dataloader initialised successfully')
        print(self.length)

    def __getitem__(self, index):
        # Open the Pickle file for reading in binary mode
        with open(self.file_list[index], 'rb') as file:
            # Load the data from the Pickle file
            loaded_data = pickle.load(file)
        image_raw = loaded_data["raw_img_pil_rgb"]
        image_recons = loaded_data["recons_img_pil_rgb"]
        image_double_recons = loaded_data["double_recons_img_pil_rgb"]
        feature_raw = loaded_data["raw_feature"]
        feature_recons = loaded_data["recons_feature"]
        feature_double_recons = loaded_data["double_recons_feature"]
        if self.transform:
            image_raw = self.transform(image_raw)
            image_recons = self.transform(image_recons)

        return image_raw, image_recons, image_double_recons, feature_raw, feature_recons, feature_double_recons

    def __len__(self):
        return self.length