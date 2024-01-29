import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device, "//", torch.cuda.get_device_name(0))