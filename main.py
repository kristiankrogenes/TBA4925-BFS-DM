import argparse
import torch
import kornia.augmentation as KA 

from dataset import BFSDataset
from model.BFSDiffusionModel import BFSDiffusionModel

from utils.utils import check_parameters_for_training

class CFG():

    def __init__(self, epochs, size, batch_size, timesteps, model_path, save_model):
        if check_parameters_for_training(epochs, size, batch_size, timesteps, model_path):
            self.epochs = epochs
            self.TARGET_SIZE = (size, size)
            self.batch_size = batch_size
            self.timesteps = timesteps
            self.model_path = model_path
            self.save_model = False if save_model==None else save_model
        else:
            raise ValueError("Something went wrong.") 

if __name__ == "__main__":

    # // ARGUMENT PARSER // ============================================================================
    parser = argparse.ArgumentParser(description="Parameters to train a Diffusion Model.")
    parser.add_argument('--epochs', type=int, help="How many epochs the model will train on.")
    parser.add_argument('--size', type=int, help="The size of the images during training.")
    parser.add_argument('--batch_size', type=int, help="The batch size")
    parser.add_argument('--timesteps', type=int, help="The number of timesteps.")
    parser.add_argument('--model_path', type=str, help="The path of a model to continue training on.")
    parser.add_argument('--save_model', type=bool, help="Should the model be saved or not.")
    args = parser.parse_args()
    # ==================================================================================================

    cfg = CFG(args.epochs, args.size, args.batch_size, args.timesteps, args.model_path, args.save_model)

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
                            num_timesteps=cfg.timesteps,
                            checkpoint=cfg.model_path)
    
    bfs.train(epochs=cfg.epochs, save_model=cfg.save_model)



