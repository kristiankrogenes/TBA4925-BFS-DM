import argparse


from model.BFSDiffusionModel import BFSDiffusionModel

import kornia.augmentation as KA 
from dataset import BFSDataset
from utils.utils import transform_model_output_to_image
import torch

if __name__ == "__main__":

    # // ARGUMENT PARSER // ============================================================================
    parser = argparse.ArgumentParser(description="Parameters to run inference of a Diffusion Model.")
    parser.add_argument('--epochs', type=str, help="Which epoch the model is trained on.")
    args = parser.parse_args()
    # ==================================================================================================

    checkpoint_path = f"./checkpoints/UNet_128x128_bs10_t100_v1_e{args.epochs}.ckpt"

    T = [
        KA.Resize(size=(128, 128), antialias=True),
        KA.RandomHorizontalFlip(),
        # KA.ToTensor(), # Scales data into [0,1] 
        # KA.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
    ]

    pred_ds = BFSDataset(
        'data/processed/predictions',
        transforms=T,
        paired=False,
        return_pair=False
    )


    bfs = BFSDiffusionModel(num_timesteps=100, checkpoint=checkpoint_path)

    for k in range(len(pred_ds)):

        input_image = transform_model_output_to_image(pred_ds[10])
        input_image.save(f"./input_v1_{k}.png", format="PNG")

        generated_tensor = bfs(batch_input=pred_ds[10].unsqueeze(0))

        generated_tensor = generated_tensor.detach().cpu()

        generated_image = transform_model_output_to_image(generated_tensor[0])

        generated_image.save(f"./output_v1_{k}.png", format="PNG")