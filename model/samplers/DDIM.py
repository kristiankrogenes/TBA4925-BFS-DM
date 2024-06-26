import torch
from torch import nn

from .beta_schedules import *
    
class DDIMSampler(nn.Module):
    
    def __init__(self,
                num_timesteps=200,
                train_timesteps=1000,
                clip_sample=False,
                schedule='linear'
                ):
        
        super().__init__()
        
        self.num_timesteps=num_timesteps
        self.train_timesteps=train_timesteps
        self.ratio=self.train_timesteps // self.num_timesteps
        self.final_alpha_cumprod=torch.tensor([1.0])
        self.clip_sample=clip_sample
        self.schedule=schedule
        
        self.register_buffer('betas',get_beta_schedule(self.schedule,self.train_timesteps))
        self.register_buffer('betas_sqrt',self.betas.sqrt())
        self.register_buffer('alphas',1-self.betas)
        self.register_buffer('alphas_cumprod',torch.cumprod(self.alphas,0))
        self.register_buffer('alphas_cumprod_sqrt',self.alphas_cumprod.sqrt())
        self.register_buffer('alphas_one_minus_cumprod_sqrt',(1-self.alphas_cumprod).sqrt())
        self.register_buffer('alphas_sqrt',self.alphas.sqrt())
        self.register_buffer('alphas_sqrt_recip',1/(self.alphas_sqrt))

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    @torch.no_grad()
    def forward(self,*args,**kwargs):   
        return self.step(*args,**kwargs)
    
    @torch.no_grad()
    def step(self, x_t, t, z_t, ratio, eta=0):
        """
            Given approximation of noise z_t in x_t predict x_(t-1)
        """
        # assert (t<self.num_timesteps).all()
        b,c,h,w=z_t.shape
        # device=z_t.device
        
        
        # compute steps based on the ratio (number of DDPM steps to take in each DDIM)
        t=t*int(ratio)
        t_prev=t-int(ratio)
        
        alphas_cumprod = self.alphas_cumprod.to(self.device)
        alphas_cumprod_sqrt = self.alphas_cumprod_sqrt.to(self.device)

        # print("#", t, self.ratio, t_prev)
        # parameters        
        alpha_cumprod_prev = alphas_cumprod[t_prev].where(t_prev.ge(0), self.final_alpha_cumprod.to(self.device)) # >= 0
        alpha_cumprod_prev = alpha_cumprod_prev.view(b,1,1,1)
        alpha_cumprod_prev_sqrt = alpha_cumprod_prev.sqrt() #alphas_cumprod_sqrt[t_prev]
        
        # estimate origin
        x_0_pred = self.estimate_origin(x_t, t, z_t)

        if self.clip_sample:
            x_0_pred=torch.clamp(x_0_pred, -1, 1)
            
        std_dev_t = eta * self.estimate_std(t, t_prev).view(b, 1, 1, 1)
        x_0_grad = (1 - alpha_cumprod_prev - std_dev_t**2).sqrt() * z_t

        prev_sample = alpha_cumprod_prev_sqrt * x_0_pred + x_0_grad

        if t > 0:          
            noise = torch.randn(z_t.shape, dtype=z_t.dtype).to(self.device)               
            prev_sample = prev_sample + std_dev_t * eta * noise

        return prev_sample
    
    def estimate_std(self, t, t_prev):
        alphas_cumprod = self.alphas_cumprod.to(self.device)

        alpha_cumprod = alphas_cumprod[t]
        alpha_cumprod_prev = alphas_cumprod[t_prev].where(t_prev.gt(0), self.final_alpha_cumprod.to(alpha_cumprod.device))
        one_minus_alpha_cumprod = 1 - alpha_cumprod
        one_minus_alpha_cumprod_prev = 1 - alpha_cumprod_prev

        var = (one_minus_alpha_cumprod_prev / one_minus_alpha_cumprod) * (1 - alpha_cumprod / alpha_cumprod_prev)

        return var.sqrt()
    
    def estimate_origin(self, x_t, t, z_t):
        alphas_cumprod = self.alphas_cumprod.to(self.device)
        alphas_one_minus_cumprod_sqrt = self.alphas_one_minus_cumprod_sqrt.to(self.device)

        alpha_cumprod = alphas_cumprod[t].view(z_t.shape[0], 1, 1, 1)
        alpha_one_minus_cumprod_sqrt = alphas_one_minus_cumprod_sqrt[t]
        return (x_t - alpha_one_minus_cumprod_sqrt * z_t) / alpha_cumprod.sqrt()