import torch
from torch.utils.data import DataLoader

from model.DiffusionModel import DiffusionModel

class BFSDiffusionModel():
        
    def __init__(self,
                dataset=None,
                image_size=None,
                batch_size=None,
                valid_dataset=None,
                schedule=None,
                sampler=None,
                condition_type=None, 
                parameterization=None, 
                num_timesteps=None,
                lr=1e-3,
                checkpoint=None):
        
        super().__init__()
        
        print("\n------- Initializing BFS-DiffusionModel -------")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("\nDevice:", self.device, "//", torch.cuda.get_device_name(0))

        self.dataset = dataset
        self.lr = lr
        self.image_size = image_size
        self.batch_size = batch_size

        self.optimizer = "ADAM"
        self.loss = "loss"
        
        input_channels = 1
        if condition_type == "v1":
            input_channels += 1
        elif condition_type == "v2":
            input_channels += 0
        elif condition_type == "v3":
            input_channels += 0
            
        self.model = DiffusionModel(batch_size=self.batch_size, 
                                    image_size=self.image_size,
                                    generated_channels=input_channels,
                                    schedule=schedule,
                                    sampler=sampler,
                                    parameterization=parameterization,
                                    condition_type=condition_type,
                                    num_timesteps=num_timesteps, 
                                    checkpoint=checkpoint, 
                                    device=self.device)
    
    @torch.no_grad()
    def forward(self, batch_input, orto_input, ts):
        return self.model(batch_input, orto_input, ts)
    
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
    
    def train(self, start_epoch, epochs, model_name=None):
        print("\nStarting training...")
        data_loader = self.dataloader()
        self.model.train(start_epoch, epochs, data_loader, model_name=model_name)
    
    def __call__(self, batch_input=None, orto_input=None, ts=None):
        return self.forward(batch_input, orto_input, ts)