import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

import kornia.augmentation as KA 


"""
    Building Footprint Segmentations (BFS) Dataset
"""
class BFSDataset(Dataset):

    def __init__(self,
                root_dir,
                type="train",
                transforms=None):
        
        self.root_dir = root_dir
        self.type = type
        self.transforms = transforms

        if self.transforms is not None:
            
            data_keys=['input']

            self.input_T=KA.container.AugmentationSequential(
                *self.transforms,
                data_keys=data_keys,
                same_on_batch=False
            )   
        
        supported_formats=['png']
        self.label_files = [el for el in os.listdir(os.path.join(self.root_dir, "original/labels/", self.type)) if el.split('.')[-1] in supported_formats]
        self.pred_files = [el for el in os.listdir(os.path.join(self.root_dir, "processed/predictions/", self.type)) if el.split('.')[-1] in supported_formats]

    def __len__(self):
        return len(self.label_files)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()            
        
        label_img_name = os.path.join("data/original/labels/", self.type, self.label_files[idx])
        pred_img_name = os.path.join("data/processed/predictions/", self.type, self.pred_files[idx])

        label_image_array = np.asarray(Image.open(label_img_name))
        label_image_array_writable = label_image_array.copy()
        label_image_tensor = torch.FloatTensor(label_image_array_writable)

        pred_image_array = np.asarray(Image.open(pred_img_name))
        pred_image_array_writable = pred_image_array.copy()
        pred_image_tensor = torch.FloatTensor(pred_image_array_writable)


        if self.transforms is not None:
            label_image = self.input_T(label_image_tensor)[0]
            label_image = label_image / 255
            label_image = (label_image.clip(0, 1).mul_(2)).sub_(1)

            pred_image = self.input_T(pred_image_tensor)[0]
            pred_image = pred_image / 255
            pred_image = (pred_image.clip(0, 1).mul_(2)).sub_(1)

            return label_image, pred_image # [C, H, W]
        else:
            return label_image_tensor, pred_image_tensor