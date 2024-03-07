import re
import argparse
import kornia.augmentation as KA 

from dataset import BFSDataset
from model.BFSDiffusionModel import BFSDiffusionModel

from utils.utils import check_parameters_for_training

class CFG():

    def __init__(self, epochs, size, batch_size, timesteps, parameterization, model_path=None, backbone=False, save_model=None, default=False):

        if default:
            self.start_epoch = 1
            self.epochs = 1000
            self.TARGET_SIZE = (128, 128)
            self.batch_size = 8
            self.timesteps = 100
            self.parameterization = "eps"
            self.model_path = None
            self.save_model = save_model

        elif check_parameters_for_training(epochs, size, batch_size, timesteps, parameterization, model_path, save_model):
            self.start_epoch = 1 if backbone else int(re.search(r'e(\d+)\.ckpt', model_path).group(1))
            self.epochs = epochs
            self.TARGET_SIZE = (size, size)
            self.batch_size = batch_size
            self.timesteps = timesteps
            self.parameterization = parameterization
            self.model_path = model_path
            self.save_model = save_model
            
        else:
            raise ValueError("Something went wrong.")

if __name__ == "__main__":

    # // ARGUMENT PARSER // ============================================================================
    parser = argparse.ArgumentParser(description="Parameters to train a Diffusion Model.")
    parser.add_argument('--epochs', type=int, help="How many epochs the model will train on.")
    parser.add_argument('--size', type=int, help="The size of the images during training.")
    parser.add_argument('--batch_size', type=int, help="The batch size")
    parser.add_argument('--timesteps', type=int, help="The number of timesteps.")
    parser.add_argument('--parameterization', type=str, help="What type of parameterization to use during training.")
    parser.add_argument('--model_path', type=str, help="The path of a model to continue training on.")
    parser.add_argument('--backbone', type=str, help="Is it a backbone.")
    parser.add_argument('--save_model', type=str, help="Should the model be saved or not. Write the name of your new model.")
    parser.add_argument('--default', type=bool, help="Use the deafult config parameters.")
    args = parser.parse_args()
    # ==================================================================================================

    cfg = CFG(args.epochs, args.size, args.batch_size, args.timesteps, args.parameterization, args.model_path, args.backbone, args.save_model, args.default)

    T = [
        KA.Resize(size=cfg.TARGET_SIZE, antialias=True),
        KA.RandomHorizontalFlip() 
    ]

    train_dataset = BFSDataset(
        root_dir="data",
        type="train",
        transforms=T
    )

    val_dataset = BFSDataset(
        root_dir="data",
        type="val",
        transforms=T
    )

    bfs = BFSDiffusionModel(dataset=train_dataset,
                            batch_size=cfg.batch_size,
                            image_size=cfg.TARGET_SIZE,
                            schedule="linear",
                            num_timesteps=cfg.timesteps,
                            checkpoint=cfg.model_path)
    
    bfs.train(start_epoch=cfg.start_epoch, epochs=cfg.epochs, parameterization=cfg.parameterization, save_model=cfg.save_model)