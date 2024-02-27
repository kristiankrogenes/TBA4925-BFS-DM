import os
from torchvision import transforms
import numpy as np

def transform_model_output_to_image(out_tensor):

    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    return reverse_transforms(out_tensor)

def check_parameters_for_training(epochs, size, batch_size, timesteps, model_path):
    if epochs == None or epochs < 1:
        raise ValueError("Number of epochs must be given and be a positive number.")
    elif size < 1:
        raise ValueError("Size must be positive.")
    elif batch_size < 1:
        raise ValueError("Batch size must be positive.")
    elif timesteps < 1:
        raise ValueError("Number of timesteps must be positive.")    
    elif not model_path==None and not os.path.exists(model_path):
        raise ValueError("Model requested does not exist.")
    return True