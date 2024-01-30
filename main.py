import torch
from dataset import BFSDataset



if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device, "//", torch.cuda.get_device_name(0))

    label_ds = BFSDataset(
        'data/original/labels',
        transforms=None,
        paired=False,
        return_pair=False
    )

    pred_ds = BFSDataset(
        'data/processed/predictions',
        transforms=None,
        paired=False,
        return_pair=False
    )