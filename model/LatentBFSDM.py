import torch
from torch.utils.data import DataLoader

from model.DiffusionModel import DiffusionModel
from model.latent_autoencoder import AutoEncoder

class LatentBFSDM():
        
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

        self.latent_scale_factor = torch.tensor(0.1)
        self.auto_encoder = AutoEncoder()

        with torch.no_grad():
            latent_tensor = self.auto_encoder.encode(torch.ones(1, 3, self.image_size[0], self.image_size[1]))
            print(latent_tensor.shape)
            self.latent_dim = latent_tensor.shape[1]

            
        self.model = DiffusionModel(batch_size=self.batch_size, 
                                    image_size=self.image_size,
                                    generated_channels=self.latent_dim,
                                    schedule=schedule,
                                    sampler=sampler,
                                    parameterization=parameterization,
                                    condition_type=condition_type,
                                    num_timesteps=num_timesteps, 
                                    checkpoint=checkpoint, 
                                    device=self.device)
    
    @torch.no_grad()
    def forward(self, batch_input, orto_input):

        batch_input = batch_input.expand(-1, 3, -1, -1)
        condition_latent = self.auto_encoder.encode(batch_input.to(self.device)).detach() * self.latent_scale_factor

        model_output_latent = self.model(condition_latent, latent_ae=[self.auto_encoder,  self.latent_scale_factor]) / self.latent_scale_factor

        return self.ae.decode(model_output_latent)
    
    def dataloader(self):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=1)
    
    def train(self, start_epoch, epochs, model_name=None):
        print("\nStarting training...")
        data_loader = self.dataloader()
        self.model.train(
            start_epoch, 
            epochs, 
            data_loader, 
            model_name=model_name, 
            latent_ae=[self.auto_encoder,  self.latent_scale_factor]
        )
    
    def __call__(self, batch_input=None, orto_input=None):
        return self.forward(batch_input, orto_input)