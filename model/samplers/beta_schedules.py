# taken from https://huggingface.co/blog/annotated-diffusion
import torch
import math

torch.pi = torch.acos(torch.zeros(1)).item() * 2

def get_beta_schedule(variant, timesteps):
    
    if variant=='cosine':
        return cosine_beta_schedule(timesteps)
    elif variant=='linear':
        return linear_beta_schedule(timesteps)
    elif variant=='quadratic':
        return quadratic_beta_schedule(timesteps)
    elif variant=='sigmoid':
        return sigmoid_beta_schedule(timesteps)
    elif variant=='softplus':
        return softplus_beta_schedule(timesteps)
    else:
        raise NotImplemented

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

def softplus_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-3, 1, timesteps)
    softplus_func = lambda x: math.log(1 + math.exp(x))
    softplus_betas = torch.tensor([softplus_func(beta) for beta in betas])
    return softplus_betas * (beta_end - beta_start) + beta_start

if __name__ == "__main__":

    cosine = cosine_beta_schedule(1000)
    linear = linear_beta_schedule(1000)
    quadratic = quadratic_beta_schedule(1000)
    sigmoid = sigmoid_beta_schedule(1000)
    softplus = softplus_beta_schedule(1000)

    betas = cosine
    betas_sqrt = betas.sqrt()
    alphas = 1 - betas
    alphas_sqrt = alphas.sqrt()
    alphas_cumprod = torch.cumprod(alphas, 0)
    alphas_cumprod_sqrt = alphas_cumprod.sqrt()
    alphas_one_minus_cumprod_sqrt = (1 - alphas_cumprod).sqrt()

    x = [i for i in range(1000)]
    y = betas.numpy()

    if True:
        import matplotlib.pyplot as plt
        from PIL import Image
        import io

        plt.plot(x, y, color='red', linestyle='-')

        plt.xlabel("Timesteps")
        plt.ylabel("Beta values")
        plt.title("Cosine Scheduler")

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)

        image = Image.open(buffer)
        
        image.save('line_plot_pil.png')