import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from model.samplers.forward_process import GaussianForwardProcess
from model.samplers.DDPM import DDPMSampler
from model.network.DMUNet import UnetConvNextBlock
from model.network.SimpleUnet import SimpleUnet
from utils.utils import transform_model_output_to_image

from PIL import Image
import numpy as np

class DiffusionModel(nn.Module):

    def __init__(self,
                generated_channels=1,
                loss_fn=F.mse_loss,
                learning_rate=1e-3,
                schedule="linear",
                num_timesteps=1000,
                sampler=None,
                checkpoint=None,
                device="cpu", 
                batch_size=None,
                image_size=None):
        
        super().__init__()

        self.device = device

        self.generated_channels = generated_channels
        self.num_timesteps = num_timesteps
        self.batch_size = batch_size
        self.image_size = image_size

        self.loss_fn = loss_fn

        self.forward_process = GaussianForwardProcess(num_timesteps=self.num_timesteps, schedule=schedule)
        self.forward_process.to(self.device)

        # self.network = UnetConvNextBlock(dim=64,
        #                                 dim_mults=(1,2,4,8),
        #                                 channels=self.generated_channels,
        #                                 out_dim=self.generated_channels,
        #                                 with_time_emb=True)
        self.network = SimpleUnet()
        self.network = self.network.to(self.device)
        
        self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=learning_rate)

        self.sampler = DDPMSampler(num_timesteps=self.num_timesteps) if sampler is None else sampler
        self.sampler = self.sampler.to(self.device)

        self.start_epoch = 1
        if checkpoint is not None:
            ckpt = torch.load(checkpoint)
            self.network.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.start_epoch = ckpt["epoch"]


    @torch.no_grad()
    def forward(self, batch_input=None):
        
        b, h, w = 1, 128, 128

        # 1. Create initial noise with Gaussian Forward Process
        if not batch_input is None:
            batch_input = batch_input.to(self.device)
            # x_t = self.forward_process(batch_input, self.num_timesteps-1)
            for i in range(self.num_timesteps):
                x_t = self.forward_process.step(batch_input, i)
        else:
            x_t = torch.randn([b, self.generated_channels, h, w], device=self.device)

        # 2. Denoising image, step-by-step
        it = reversed(range(0, self.num_timesteps))
        for i in tqdm(it, desc='Diffusion Sampling', total=self.num_timesteps):

            t = torch.full((b,), i, device=self.device, dtype=torch.long)

            if (i % 10 == 0 or i in [100, 99, 98, 97, 96, 95, 94, 93, 92, 91]) and False:
                xt_tensor = x_t.detach().cpu()
                outs = transform_model_output_to_image(xt_tensor[0])
                outs.save(f"./ii/out{i}.png", format="PNG")

            z_t = self.network(x_t, t)          # prediction of noise
            x_t = self.sampler(x_t, t, z_t)     # prediction of next state
            
        return torch.clamp(x_t, -1.0, 1.0)

    def train(self, epochs, data_loader, save_model=False):

        for epoch in range(self.start_epoch, epochs+1):
            loss_epoch = []
            for batch in tqdm(data_loader, desc=f"Epoch {epoch} - Diffusion Training"):

                # b, c, h, w = batch.shape
                label, pred = batch
                # print(label.shape, pred.shape)
                
                b, c, h, w = label.shape
                
                label = label.to(self.device)
                pred = pred.to(self.device)

                t = torch.randint(0, self.num_timesteps, (b,), device=self.device).long()

                label_xt, label_noise = self.forward_process(label, t, return_noise=True)
                pred_xt, pred_noise = self.forward_process(pred, t, return_noise=True)

                model_input = pred_xt.to(self.device)
                # noise = noise.to(self.device)

                noise_prediction = self.network(model_input, t)

                loss = self.loss_fn(noise_prediction, label_noise)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_epoch.append(loss.item())
            
            avg_loss_epoch = sum(loss_epoch) / len(loss_epoch)
            print("Epoch", epoch, "// Loss", avg_loss_epoch)

            if save_model:
                if epoch % 100 == 0:
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.network.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                    }
                    new_checkpoint_path = f"./checkpoints/BaselineLabelPred2/UNet_{self.image_size[0]}x{self.image_size[1]}_bs{self.batch_size}_t{self.num_timesteps}_v2"
                    torch.save(checkpoint, f"{new_checkpoint_path}_e{epoch}.ckpt")
                    print(f"Model saved at epoch {epoch}")



