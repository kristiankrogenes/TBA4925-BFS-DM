import torch
from torch.utils.data import DataLoader

from model.DiffusionModel import DiffusionModel

class BFSDiffusionModel():
        
    def __init__(self,
                train_dataset=None,
                batch_size=None,
                valid_dataset=None,
                num_timesteps=None,
                lr=1e-3,
                checkpoint=None):
        
        super().__init__()

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.lr = lr
        self.batch_size = batch_size

        self.optimizer = "ADAM"
        self.loss = "loss"

        self.model = DiffusionModel(num_timesteps=num_timesteps, checkpoint=checkpoint)
    
    @torch.no_grad()
    def forward(self, *args, **kwargs):
        # return self.output_T(self.model(*args, **kwargs))
        return self.model(*args, **kwargs)
    
    def input_T(self, input):
        # By default, let the model accept samples in [0,1] range, and transform them automatically
        return (input.clip(0, 1).mul_(2)).sub_(1)
    
    def output_T(self, input):
        # Inverse transform of model output from [-1,1] to [0,1] range
        return (input.add_(1)).div_(2)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=1)
    
    def val_dataloader(self):
        if self.valid_dataset is not None:
            return DataLoader(self.valid_dataset,
                              batch_size=self.batch_size,
                              shuffle=False,
                              num_workers=1)
        else:
            return None
    
    def train(self, epochs, checkpoint_path=None):
        data_loader = self.train_dataloader()
        self.model.train(epochs, data_loader, checkpoint_path=checkpoint_path)
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)