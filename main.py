import argparse
import kornia.augmentation as KA 

from dataset import BFSDataset
from model.BFSDiffusionModel import BFSDiffusionModel
from model.LatentBFSDM import LatentBFSDM

import model.config
import model.configV2


if __name__ == "__main__":

    # // ARGUMENT PARSER // ============================================================================
    parser = argparse.ArgumentParser(description="Parameters to train a Diffusion Model.")
    # parser.add_argument("--epochs", type=int, help="How many epochs the model will train on.")
    # parser.add_argument("--size", type=int, help="The size of the images during training.")
    # parser.add_argument("--batch_size", type=int, help="The batch size")
    # parser.add_argument("--timesteps", type=int, help="The number of timesteps.")
    # parser.add_argument("--parameterization", type=str, help="What type of parameterization to use during training.")
    # parser.add_argument("--model_path", type=str, help="The path of a model to continue training on.")
    # parser.add_argument("--backbone", type=str, help="Is it a backbone.")
    # parser.add_argument("--save_model", type=str, help="Should the model be saved or not. Write the name of your new model.")
    # parser.add_argument("--default", type=bool, help="Use the deafult config parameters.")
    parser.add_argument("--config_name", type=str, help="What config to train.")
    args = parser.parse_args()
    # ==================================================================================================

    cfg = None
    for config in model.configV2.configs:
        if args.config_name==config.model_name:
            cfg = config
    if cfg is None:
        raise ValueError("This config does not exist.")

    T=[
        # KA.RandomCrop((2*64,2*64)),
        KA.Resize(cfg.TARGET_SIZE, antialias=True),
        # KA.Resize((512, 512), antialias=True),
        # KA.RandomVerticalFlip()
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

    # bfs = BFSDiffusionModel(dataset=train_dataset,
    #                         batch_size=cfg.batch_size,
    #                         image_size=cfg.TARGET_SIZE,
    #                         parameterization=cfg.parameterization,
    #                         condition_type=cfg.condition_type,
    #                         schedule=cfg.schedule,
    #                         # sampler="DDIM",
    #                         num_timesteps=cfg.timesteps,
    #                         checkpoint=cfg.model_path)
    
    bfs = LatentBFSDM(dataset=train_dataset,
                    batch_size=cfg.batch_size,
                    image_size=cfg.TARGET_SIZE,
                    parameterization=cfg.parameterization,
                    condition_type=cfg.condition_type,
                    schedule=cfg.schedule,
                    num_timesteps=cfg.timesteps,
                    checkpoint=cfg.model_path)
    
    bfs.train(start_epoch=cfg.start_epoch, epochs=cfg.epochs, model_name=cfg.model_name)

    # from model.samplers.forward_process import GaussianForwardProcess
    # from torch.utils.data import DataLoader
    # fp = GaussianForwardProcess(1000, "softplus")

    # data_batch = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

    # for bi, batch in enumerate(data_batch):
    #     if bi == 3:
    #         label, x_0, orto = batch
    #         print(label.shape, x_0.shape, orto.shape)
    #         x_t = label
    #         for ti in range(1000):
    #             x_t = fp.step(x_t, ti, return_noise=False, sample_path="3392_10912")
    #         break