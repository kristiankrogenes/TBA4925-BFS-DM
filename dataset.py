import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

"""
    Building Footprint Segmentations (BFS) Dataset
"""
class BFSDataset(Dataset):

    def __init__(self,
                root_dir,
                transforms=None,
                paired=True,
                return_pair=False):
        
        self.root_dir = root_dir
        self.transforms = transforms
        self.paired=paired
        self.return_pair=return_pair
        
        # set up transforms
        # if self.transforms is not None:
        #     if self.paired:
        #         data_keys=2*['input']
        #     else:
        #         data_keys=['input']

        #     self.input_T=KA.container.AugmentationSequential(
        #         *self.transforms,
        #         data_keys=data_keys,
        #         same_on_batch=False
        #     )   
        
        # check files
        supported_formats=['png']        
        self.files=[el for el in os.listdir(self.root_dir) if el.split('.')[-1] in supported_formats]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()            

        img_name = os.path.join(self.root_dir, self.files[idx])
        image_array = np.asarray(Image.open(img_name)) / 255
        image_tensor = torch.FloatTensor(image_array)

        assert image_tensor.shape == torch.Size((512, 512)), "Image tensor does not have the correct shape."

        # if self.paired:
        #     c,h,w=image.shape
        #     slice=int(w/2)
        #     image2=image[:,:,slice:]
        #     image=image[:,:,:slice]
        #     if self.transforms is not None:
        #         out = self.input_T(image, image2)
        #         image=out[0][0]
        #         image2=out[1][0]
        # elif self.transforms is not None:
        #     image = self.input_T(image)[0]

        # if self.return_pair:
        #     return image2, image
        # else:
        #     return image_tensor
        
        return image_tensor