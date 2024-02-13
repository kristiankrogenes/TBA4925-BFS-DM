import torch
from torch.utils.data import DataLoader

from model.DiffusionModel import DiffusionModel

class BFSDiffusionModel():
        
    def __init__(self,
                dataset=None,
                image_size=None,
                batch_size=None,
                valid_dataset=None,
                num_timesteps=None,
                lr=1e-3,
                checkpoint=None):
        
        super().__init__()
        
        print("Initializing BFS-DifussinModel -------")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("\nDevice:", self.device, "//", torch.cuda.get_device_name(0), "\n")

        self.dataset = dataset
        self.lr = lr
        self.image_size = image_size
        self.batch_size = batch_size

        self.optimizer = "ADAM"
        self.loss = "loss"

        self.model = DiffusionModel(batch_size=self.batch_size, 
                                    image_size=self.image_size, 
                                    num_timesteps=num_timesteps, 
                                    checkpoint=checkpoint, 
                                    device=self.device)
    
    @torch.no_grad()
    def forward(self, batch_input):
        return self.model(batch_input)
    
    def input_T(self, input):
        # By default, let the model accept samples in [0,1] range, and transform them automatically
        return (input.clip(0, 1).mul_(2)).sub_(1)
    
    def output_T(self, input):
        # Inverse transform of model output from [-1,1] to [0,1] range
        return (input.add_(1)).div_(2)
    
    def dataloader(self):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=1)
    
    def train(self, epochs, save_model=False):
        data_loader = self.dataloader()
        self.model.train(epochs, data_loader, save_model=save_model)
    
    def __call__(self, batch_input=None):
        return self.forward(batch_input)