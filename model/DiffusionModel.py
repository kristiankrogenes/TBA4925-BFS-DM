import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from model.samplers.forward_process import GaussianForwardProcess
from model.samplers.DDPM import DDPMSampler
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

        self.sampler = DDPMSampler(num_timesteps=self.num_timesteps) if sampler is None else sampler

        if checkpoint is not None:
            ckpt = torch.load(checkpoint)
            self.network.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])


    @torch.no_grad()
    def forward(self, batch_input=None, orto_input=None):

        # if not batch_input is None:
        #     x_t = batch_input.to(self.device)
        #     x_t = self.forward_process(batch_input, self.num_timesteps)
        # else:
        #     x_t = torch.randn([1, self.generated_channels, self.image_size, self.image_size], device=self.device)
        b, c, h, w = batch_input.shape
        condition = batch_input.to(self.device)
        if not orto_input == None:
            orto = orto_input.to(self.device)

        x_t = torch.randn([b, c, h, w], device=self.device)

        it = reversed(range(0, self.num_timesteps))
        for i in tqdm(it, desc='Diffusion Sampling', total=self.num_timesteps):

            t = torch.full((1,), i, device=self.device, dtype=torch.long)

            if orto_input==None:
                model_input = torch.cat([x_t, condition], 1).to(self.device)
            else:
                model_input = torch.cat([x_t, condition, orto], 1).to(self.device)

            z_t = self.network(model_input, t)

            if self.parameterization == "eps":
                x_t = self.sampler(x_t, t, z_t)
            elif self.parameterization == "x0":
                x_t = z_t
            elif self.parameterization == "v":
                raise NotImplementedError(f"Parameterization {self.parameterization} not yet supported")
            else:
                raise ValueError(f"Parameterization {self.parameterization} not valid.")
           
            # -------------------------------------------------------------------------------
            if i % 10 == 0 and False:
                zt_tensor = z_t.detach().cpu()
                clamped_zt = torch.clamp(zt_tensor, min=-1.0, max=1.0)
                outs = transform_model_output_to_image(clamped_zt[0])
                outs.save(f"./ii/out{i}.png", format="PNG")
            # -------------------------------------------------------------------------------
            
        return torch.clamp(x_t, min=-1.0, max=1.0)

    def train(self, start_epoch, epochs, data_loader, model_name=None):

        for epoch in range(start_epoch, epochs+1):
            loss_epoch = []
            for bi, batch in enumerate(tqdm(data_loader, desc=f"Epoch {epoch} - Diffusion Training")):

                label, x_0, orto = batch
                
                b, c, h, w = label.shape
                
                label = label.to(self.device)
                x_0 = x_0.to(self.device)
                orto = orto.to(self.device)

                t = torch.randint(0, self.num_timesteps, (b,), device=self.device).long()

                x_t, noise = self.forward_process(label, t, return_noise=True)
                x_t = x_t.to(self.device)

                fade_ts = t.float() / self.num_timesteps
                fade_x0_broadcast = fade_ts.view(-1, 1, 1, 1)
                fade_orto_broadcast = fade_ts.view(-1, 1, 1, 1)
                x_0 = x_0 * fade_x0_broadcast
                orto = orto * fade_orto_broadcast
                
                if self.generated_channels == 2:
                    model_input = torch.cat([x_t, x_0], 1).to(self.device)
                elif self.generated_channels == 5:
                    model_input = torch.cat([x_t, x_0, orto], 1).to(self.device)

                z_t = self.network(model_input, t)

                if self.parameterization == "eps":
                    target = noise
                elif self.parameterization == "x0":
                    target = label
                elif self.parameterization == "v":
                    raise NotImplementedError(f"Parameterization {self.parameterization} not yet supported")
                else:
                    raise ValueError(f"Training parameterization {self.parameterization} not valid.")
                
                loss = self.loss_fn(target, z_t)
                

                # -------------------------------------------------------------------------------
                if epoch % 50 == 0 and False:
                    prediction_image = transform_model_output_to_image(clamped_zt[0].detach().cpu())
                    if not os.path.exists(f"ii/tester/epoch_{epoch}"):
                        os.makedirs(f"ii/tester/epoch_{epoch}")
                    prediction_image.save(f"ii/tester/epoch_{epoch}/training_prediction_from_batch_{bi}.png", format="PNG") 
                # -------------------------------------------------------------------------------
                    
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_epoch.append(loss.item())
            
            avg_loss_epoch = sum(loss_epoch) / len(loss_epoch)
            print("Epoch", epoch, "// Loss", avg_loss_epoch)

            metrics_path = f"./outputs/metrics/loss_metrics_{model_name}.csv"
            if not os.path.exists(metrics_path):
                # os.makedirs(metrics_path)
                with open(metrics_path, "w", newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)

            add_row_to_csv(metrics_path, [epoch, avg_loss_epoch])

            if model_name:
                if epoch % 50 == 0:
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

    



