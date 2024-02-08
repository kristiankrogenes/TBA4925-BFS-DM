import numpy as np
from torchvision import transforms

from model.BFSDiffusionModel import BFSDiffusionModel

def transform_model_output_to_image(out_tensor):

    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    return reverse_transforms(out_tensor)

if __name__ == "__main__":

    checkpoint_path = "./checkpoints/UNet_128x128_bs10_t100_v1_e1000.ckpt"

    bfs = BFSDiffusionModel(num_timesteps=100, checkpoint=checkpoint_path)

    for k in range(1):
        generated_tensor = bfs()
        generated_tensor = generated_tensor.detach().cpu()

        generated_image = transform_model_output_to_image(generated_tensor[0])

        generated_image.save(f"./output_v1_{k}.png", format="PNG")