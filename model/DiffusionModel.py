import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from model.samplers.forward_process import GaussianForwardProcess
from model.samplers.DDPM import DDPMSampler
from model.network.DMUNet import UnetConvNextBlock
from model.network.SimpleUnet import SimpleUnet

from PIL import Image
import numpy as np

from torchvision import transforms
def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])
    return reverse_transforms(image)

class DiffusionModel(nn.Module):

    def __init__(self,
                generated_channels=1,              
                loss_fn=F.mse_loss,
                learning_rate=1e-3,
                schedule='linear',
                num_timesteps=1000,
                sampler=None,
                checkpoint=None):
        
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.generated_channels = generated_channels
        self.num_timesteps = num_timesteps

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
        
        # b, c, h, w = batch_input.shape
        b, h, w = 1, 128, 128
        # 1. Create initial noise with Gaussian Forward Process
        x_t = torch.randn([b, self.generated_channels, h, w], device=self.device)
        # x_t = self.forward_process(batch_input, self.num_timesteps-1)

        # 2. Denoising image, step-by-step
        it = reversed(range(0, self.num_timesteps))
        for i in tqdm(it, desc='Diffusion sampling', total=self.num_timesteps):

            t = torch.full((b,), i, device=self.device, dtype=torch.long)

            # model_input = torch.cat([x_t, batch_input], 1).to(device)
            # model_input = x_t.to(self.device)

            z_t = self.network(x_t, t) # prediction of noise

            x_t = self.sampler(x_t, t, z_t) # prediction of next state
            
            # generated_img = torch.clamp(x_t, -1.0, 1.0)
            if i % 10 == 0:
                generated_img = torch.clamp(x_t, -1.0, 1.0)
                xt_tensor = generated_img.detach().cpu()
                outs = show_tensor_image(xt_tensor[0])
                # xt = Image.fromarray((xt_tensor[0][0] * 255).numpy().astype(np.uint8))
                outs.save(f"./ii/out{i}.png", format="PNG")
            
        return torch.clamp(x_t, -1.0, 1.0)

    def train(self, epochs, data_loader, checkpoint_path=None):

        for epoch in range(self.start_epoch, epochs+1):
            loss_epoch = []
            for batch in tqdm(data_loader, desc=f"Epoch {epoch} - Diffusion Training"):

                b, c, h, w = batch.shape
                
                batch = batch.to(self.device)

                # t = torch.full((b,), self.num_timesteps, device=self.device, dtype=torch.long)
                # t = torch.full((b,), self.num_timesteps, device=self.device, dtype=torch.long)
                t = torch.randint(0, self.num_timesteps, (b,), device=self.device).long()

                x_t, noise = self.forward_process(batch, t, return_noise=True)

                mi = x_t.to(self.device)
                noise = noise.to(self.device)

                noise_pred = self.network(mi, t)

                loss = self.loss_fn(noise_pred, noise)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_epoch.append(loss.item())
            
            avg_loss_epoch = sum(loss_epoch) / len(loss_epoch)
            print("Epoch", epoch, "//", avg_loss_epoch)

            if checkpoint_path is not None:
                if epoch % 100 == 0:
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.network.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                    }
                    torch.save(checkpoint, f"{checkpoint_path}_e{epoch}.ckpt")
                    print(f"Model saved at epoch {epoch}")



