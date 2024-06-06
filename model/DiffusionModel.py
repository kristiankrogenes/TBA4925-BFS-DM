import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime

from model.samplers.forward_process import GaussianForwardProcess
from model.samplers.DDPM import DDPMSampler
from model.samplers.DDIM import DDIMSampler
from model.network.DMUNet import UnetConvNextBlock
from model.network.SimpleUnet import SimpleUnet
from utils.utils import transform_model_output_to_image, add_row_to_csv

class DiffusionModel(nn.Module):

    def __init__(self,
                generated_channels=None,
                loss_fn=F.mse_loss,
                learning_rate=1e-4,
                schedule=None,
                parameterization=None, 
                condition_type=None,
                num_timesteps=None,
                sampler=None,
                checkpoint=None,
                device="cpu", 
                batch_size=None,
                image_size=None):
        
        super().__init__()

        self.device = device
        self.generated_channels = generated_channels
        self.parameterization = parameterization
        self.condition_type = condition_type
        self.num_timesteps = num_timesteps
        self.batch_size = batch_size
        self.image_size = image_size

        self.loss_fn = loss_fn

        self.forward_process = GaussianForwardProcess(num_timesteps=self.num_timesteps, schedule=schedule)

        self.network = UnetConvNextBlock(dim=64,
                                        dim_mults=(1, 2, 4, 8),
                                        channels=generated_channels,
                                        out_dim=1,
                                        with_time_emb=True)
        # self.network = SimpleUnet()
        self.network.to(self.device)
        
        self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=learning_rate)

        if sampler is None:
            self.sampler = DDPMSampler(num_timesteps=self.num_timesteps, schedule=schedule)
        elif sampler=="DDIM":
            self.sampler = DDIMSampler(train_timesteps=self.num_timesteps, schedule=schedule)


        if checkpoint is not None:
            ckpt = torch.load(checkpoint)
            self.network.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])


    @torch.no_grad()
    def forward(self, batch_input=None, orto_input=None, ts=None, latent_ae=None):

        b, c, h, w = batch_input.shape
        condition = batch_input.to(self.device)
        if not orto_input == None:
            orto = orto_input.to(self.device)

        x_t = torch.randn([b, c, h, w], device=self.device)        

        # self.num_timesteps = 1000
        # ts = self.num_timesteps
        # x_t = self.forward_process(condition, self.num_timesteps, return_noise=False)

        # it = reversed(range(0, self.num_timesteps))
        # for i in tqdm(it, desc='DDPM - Diffusion Sampling', total=self.num_timesteps):

        it = reversed(range(0, ts))
        for i in tqdm(it, desc='DDIM - Diffusion Sampling', total=ts):
            
            t = torch.full((1,), i, device=self.device, dtype=torch.long)

            if self.condition_type=="v1":
                model_input = torch.cat([x_t, condition], 1).to(self.device)
                z_t = self.network(model_input, time=t)
            elif self.condition_type=="v2":
                model_input = x_t + condition
                z_t = self.network(model_input, time=t)
            elif self.condition_type=="v3":
                z_t = self.network(x_t, condition, time=t*(1000/ts))
                # z_t = self.network(x_t, condition, time=t)
            else:
                # model_input = torch.cat([x_t, condition], 1).to(self.device)
                model_input = x_t
                z_t = self.network(model_input, time=t)

            if self.parameterization == "eps":
                # x_t = self.sampler(x_t, t, z_t)
                x_t = self.sampler(x_t, t, z_t, 1000/ts, eta=1.)
            elif self.parameterization == "x0":
                x_t = z_t
            elif self.parameterization == "v":
                raise NotImplementedError(f"Parameterization {self.parameterization} not yet supported")
            else:
                raise ValueError(f"Parameterization {self.parameterization} not valid.")
            
            if i%10 == 0 or i==ts-1:
                generated_tensor = torch.clamp(x_t, min=-1.0, max=1.0)
                generated_tensor = generated_tensor.detach().cpu()
                generated_image = transform_model_output_to_image(generated_tensor[0])

                generated_image.save(f"./subrev/norkart/subrev_{i}.png", format="PNG")  

        return torch.clamp(x_t, min=-1.0, max=1.0)

        # return x_t

    def train(self, start_epoch, epochs, data_loader, model_name=None, latent_ae=None):

        for epoch in range(start_epoch, epochs+1):
            epochStartTime = datetime.now()
            loss_epoch = []
            for bi, batch in enumerate(tqdm(data_loader, desc=f"Epoch {epoch} - Diffusion Training")):

                label, x_0, orto = batch
                
                b, c, h, w = label.shape
                if latent_ae:
                    label = label.expand(-1, 3, -1, -1)
                    x_0 = x_0.expand(-1, 3, -1, -1)
                    label = latent_ae[0].encode(label).detach() * latent_ae[1]      # Latent label
                    x_0 = latent_ae[0].encode(x_0).detach() * latent_ae[1]          # Latent condition
                
                label = label.to(self.device)
                x_0 = x_0.to(self.device)
                orto = orto.to(self.device)

                
                if self.condition_type == "v1":
                    t = torch.randint(0, self.num_timesteps, (b,), device=self.device).long()
                    x_t, noise = self.forward_process(label, t, return_noise=True)
                    # print(x_t.shape, noise.shape)
                    x_t = x_t.to(self.device)
                    model_input = torch.cat([x_t, x_0], 1).to(self.device)
                    z_t = self.network(model_input, time=t)
                elif self.condition_type == "v2":
                    t = torch.randint(0, self.num_timesteps, (b,), device=self.device).long()
                    x_t, noise = self.forward_process(label, t, return_noise=True)
                    x_t = x_t.to(self.device)
                    model_input = x_t + x_0
                    z_t = self.network(model_input, time=t)
                elif self.condition_type == "v3":
                    t = torch.randint(0, self.num_timesteps, (b,), device=self.device).long()
                    x_t, noise = self.forward_process(label, t, return_noise=True)
                    
                    # if latent_ae:
                    #     x_t = x_t.expand(-1, 3, -1, -1)
                    #     x_t = latent_ae[0].encode(x_t).detach() * latent_ae[1]
                    x_t = x_t.to(self.device)
                    z_t = self.network(x_t, x_0, time=t)
                else:
                    t = torch.randint(0, self.num_timesteps, (b,), device=self.device).long()
                    x_t, noise = self.forward_process(label, t, return_noise=True)
                    x_t = x_t.to(self.device)
                    z_t = self.network(x_t, time=t)


                if self.parameterization == "eps":
                    target = noise
                elif self.parameterization == "x0":
                    target = label
                elif self.parameterization == "v":
                    raise NotImplementedError(f"Parameterization {self.parameterization} not yet supported")
                else:
                    raise ValueError(f"Training parameterization {self.parameterization} not valid.")
                # print("#", target.shape, z_t.shape)
                loss = self.loss_fn(target, z_t)
                    
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_epoch.append(loss.item())
            
            avg_loss_epoch = sum(loss_epoch) / len(loss_epoch)
            print("Epoch", epoch, "// Loss", avg_loss_epoch)

            metrics_path = f"./outputs/metricsV2/loss_metrics_{model_name}.csv"
            if not os.path.exists(metrics_path):
                with open(metrics_path, "w", newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)

            epochEndTime = datetime.now()

            add_row_to_csv(metrics_path, [epoch, avg_loss_epoch, epochEndTime-epochStartTime])

            if model_name:
                if epoch % 100 == 0:
                    checkpoint = {
                        "epoch": epoch,
                        "model_state_dict": self.network.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                    }

                    folder_path = f"./checkpoints/{model_name}/"
                    checkpoint_name = f"UNet_{self.image_size[0]}x{self.image_size[1]}_bs{self.batch_size}_t{self.num_timesteps}_e{epoch}.ckpt"

                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)

                    new_checkpoint_path = os.path.join(folder_path, checkpoint_name)
                    torch.save(checkpoint, new_checkpoint_path)
                    print(f"Model saved at epoch {epoch}.")

    



