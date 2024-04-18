import re
from utils.utils import check_parameters_for_training

class CFG():

    def __init__(self, 
                epochs=None, 
                size=None, 
                batch_size=None, 
                timesteps=None, 
                parameterization=None,
                schedule=None,
                model_path=None, 
                backbone=False, 
                model_name=None):

        if check_parameters_for_training(epochs, size, batch_size, timesteps, parameterization, schedule, model_path, model_name):
            self.start_epoch = 1 if backbone else int(re.search(r'e(\d+)\.ckpt', model_path).group(1))
            self.epochs = epochs
            self.TARGET_SIZE = (size, size)
            self.batch_size = batch_size
            self.timesteps = timesteps
            self.parameterization = parameterization
            self.schedule = schedule
            self.model_path = model_path
            self.model_name = model_name
        else:
            raise ValueError("Something went wrong.")

conditionalV2 = CFG(
    epochs=10000,
    size=128,
    batch_size=16,
    timesteps=1000,
    parameterization="eps",
    schedule="linear",
    backbone=False,
    model_name="ConditionalV2",
    model_path="./checkpoints/ConditionalV2/UNet_128x128_bs16_t1000_e8300.ckpt"
)

x0Conditional = CFG(
    epochs=10000,
    size=128,
    batch_size=16,
    timesteps=1000,
    parameterization="x0",
    schedule="linear",
    backbone=False,
    model_name="x0Conditional",
    model_path="./checkpoints/x0Conditional/UNet_128x128_bs16_t1000_e8800.ckpt"
)

epsConditionalCosine = CFG(
    epochs=10000,
    size=128,
    batch_size=16,
    timesteps=1000,
    parameterization="eps",
    schedule="cosine",
    model_path="./checkpoints/epsConditionalCosine/UNet_128x128_bs16_t1000_e5000.ckpt",
    backbone=False,
    model_name="epsConditionalCosine",
)

x0ConditionalCosine = CFG(
    epochs=10000,
    size=128,
    batch_size=16,
    timesteps=1000,
    parameterization="x0",
    schedule="cosine",
    backbone=True,
    model_name="x0ConditionalCosine",
    # model_path="./checkpoints/x0Conditional/UNet_128x128_bs16_t1000_e8800.ckpt"
)

epsOrtoConditionalCosine = CFG(
    epochs=10000,
    size=128,
    batch_size=16,
    timesteps=1000,
    parameterization="eps",
    schedule="cosine",
    backbone=True,
    model_name="epsOrtoConditionalCosine",
    # model_path="./checkpoints/x0Conditional/UNet_128x128_bs16_t1000_e8800.ckpt"
)

configs = [conditionalV2, x0Conditional, epsConditionalCosine, x0ConditionalCosine, epsOrtoConditionalCosine]