import torch
import kornia.augmentation as KA 

from dataset import BFSDataset
from model.BFSDiffusionModel import BFSDiffusionModel

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\nDevice:", device, "//", torch.cuda.get_device_name(0), "\n")

    TARGET_SIZE = (128, 128)
    # T = [
    #     # KA.RandomCrop((2*CROP_SIZE,2*CROP_SIZE)),
    #     KA.Resize(size=TARGET_SIZE, antialias=True),
    #     # KA.RandomVerticalFlip()
    # ]
    T = [
        KA.Resize(size=TARGET_SIZE, antialias=True),
        KA.RandomHorizontalFlip(),
        # KA.ToTensor(), # Scales data into [0,1] 
        # KA.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
    ]
    
    label_ds = BFSDataset(
        'data/original/labels',
        transforms=T,
        paired=False,
        return_pair=False
    )

    pred_ds = BFSDataset(
        'data/processed/predictions',
        transforms=T,
        paired=False,
        return_pair=False
    )

    batch_size = 10
    epochs = 10000
    timesteps = 100

    if True:
        load_checkpoint = True
        new_checkpoint = True
        if load_checkpoint:
            checkpoint_path = "./checkpoints/UNet_128x128_bs10_t100_v1_e1000.ckpt"
        else:
            checkpoint_path = None
        if new_checkpoint:
            new_checkpoint_path = f"./checkpoints/UNet_{TARGET_SIZE[0]}x{TARGET_SIZE[1]}_bs{batch_size}_t{timesteps}_v1"
        else:
            new_checkpoint_path = None

        bfs = BFSDiffusionModel(train_dataset=label_ds,
                                batch_size=batch_size,
                                num_timesteps=timesteps,
                                checkpoint=checkpoint_path)
        
        bfs.train(epochs=epochs, checkpoint_path=new_checkpoint_path)



