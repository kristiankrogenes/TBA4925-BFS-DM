import os
import torch
from torch import nn

from model.samplers.beta_schedules import *
from utils.utils import transform_model_output_to_image
        
class GaussianForwardProcess(nn.Module):
    
    def __init__(self, num_timesteps=None, schedule=None):
        
        super().__init__()

        self.num_timesteps = num_timesteps if not num_timesteps is None else 1000
        self.schedule = schedule if not schedule is None else "linear"

        self.betas = get_beta_schedule(self.schedule, self.num_timesteps)
        self.betas_sqrt = self.betas.sqrt()
        self.alphas = 1 - self.betas
        self.alphas_sqrt = self.alphas.sqrt()
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_sqrt = self.alphas_cumprod.sqrt()
        self.alphas_one_minus_cumprod_sqrt = (1 - self.alphas_cumprod).sqrt()
     
    @torch.no_grad()
    def forward(self, x0, t, return_noise=False):

        self.alphas_cumprod_sqrt = self.alphas_cumprod_sqrt.to(x0.device)
        self.alphas_one_minus_cumprod_sqrt = self.alphas_one_minus_cumprod_sqrt.to(x0.device)
        
        b = x0.shape[0]
        

        mean = x0 * self.alphas_cumprod_sqrt[t].view(b, 1, 1, 1)
        std = self.alphas_one_minus_cumprod_sqrt[t].view(b, 1, 1, 1)
        
        noise = torch.randn_like(x0)

        xt = mean + std * noise
        
        if not return_noise:
            return xt
        else:
            return xt, noise
    
    @torch.no_grad()
    def step(self, x_t, t, return_noise=False, sample_path=None):
        
        mean = self.alphas_sqrt[t] * x_t
        std = self.betas_sqrt[t]

        noise = torch.randn_like(x_t)

        output = mean + std * noise

        if sample_path is not None:
            folder_path = os.path.join(f"./gaussian_samples/{self.schedule}/", sample_path)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            if t == 0:
                input_sample = transform_model_output_to_image(x_t[0].detach().cpu())
                input_sample.save(f"{folder_path}/input_sample_t{t}.png", format="PNG") 
            if t % 10 == 0:
                clamped_output = torch.clamp(output, min=-1.0, max=1.0)
                diffused_sample = transform_model_output_to_image(clamped_output[0].detach().cpu())
                diffused_sample.save(f"{folder_path}/diffused_sample_t{t}.png", format="PNG")           
        
        if not return_noise:
            return output
        else:
            return output, noise