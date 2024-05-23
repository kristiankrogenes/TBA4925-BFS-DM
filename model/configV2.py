import re
from utils.utils import check_parameters_for_training

class CFG2():

    def __init__(self, 
                epochs=None, 
                size=None, 
                batch_size=None, 
                timesteps=None, 
                parameterization=None,
                condition_type=None,
                schedule=None,
                model_path=None,
                model_name=None):

        if check_parameters_for_training(epochs, size, batch_size, timesteps, parameterization, condition_type, schedule, model_path, model_name):
            self.start_epoch = 1 if not model_path else int(re.search(r'e(\d+)\.ckpt', model_path).group(1))
            self.epochs = epochs
            self.TARGET_SIZE = (size, size)
            self.batch_size = batch_size
            self.timesteps = timesteps
            self.parameterization = parameterization
            self.condition_type = condition_type
            self.schedule = schedule
            self.model_path = model_path
            self.model_name = model_name
        else:
            raise ValueError("Something went wrong.")

epsLinearDDPMCondV1 = CFG2(
    epochs=10000,
    size=128,
    batch_size=16,
    timesteps=1000,
    parameterization="eps",
    condition_type="v1",
    schedule="linear",
    model_name="epsLinearDDPMCondV1",
    model_path="./checkpoints/epsLinearDDPMCondV1/UNet_128x128_bs16_t1000_e10000.ckpt"
)

epsCosineDDPMCondV1 = CFG2(
    epochs=10000,
    size=128,
    batch_size=16,
    timesteps=1000,
    parameterization="eps",
    condition_type="v1",
    schedule="cosine",
    model_name="epsCosineDDPMCondV1",
    model_path="./checkpoints/epsCosineDDPMCondV1/UNet_128x128_bs16_t1000_e10000.ckpt"
)

epsSoftplusDDPMCondV1 = CFG2(
    epochs=10000,
    size=128,
    batch_size=16,
    timesteps=1000,
    parameterization="eps",
    condition_type="v1",
    schedule="softplus",
    model_name="epsSoftplusDDPMCondV1",
    model_path="./checkpoints/epsSoftplusDDPMCondV1/UNet_128x128_bs16_t1000_e10000.ckpt"
)

x0SoftplusDDPMCondV1 = CFG2(
    epochs=10000,
    size=128,
    batch_size=16,
    timesteps=1000,
    parameterization="x0",
    condition_type="v1",
    schedule="softplus",
    model_name="x0SoftplusDDPMCondV1",
    model_path="./checkpoints/x0SoftplusDDPMCondV1/UNet_128x128_bs16_t1000_e6800.ckpt"
)

epsSoftplusDDPMCondV2 = CFG2(
    epochs=10000,
    size=128,
    batch_size=16,
    timesteps=1000,
    parameterization="eps",
    condition_type="v2",
    schedule="softplus",
    model_name="epsSoftplusDDPMCondV2",
    model_path="./checkpoints/epsSoftplusDDPMCondV2/UNet_128x128_bs16_t1000_e6600.ckpt"
)

x0SoftplusDDPMCondV2 = CFG2(
    epochs=10000,
    size=128,
    batch_size=16,
    timesteps=1000,
    parameterization="x0",
    condition_type="v2",
    schedule="softplus",
    model_name="x0SoftplusDDPMCondV2",
    model_path="./checkpoints/x0SoftplusDDPMCondV2/UNet_128x128_bs16_t1000_e6900.ckpt"
)

epsCosineDDPMCondV1_T = CFG2(
    epochs=10000,
    size=128,
    batch_size=16,
    timesteps=1000,
    parameterization="eps",
    condition_type="v1",
    schedule="cosine",
    model_name="epsCosineDDPMCondV1_T",
    # model_path="./checkpoints/epsCosineDDPMCondV1/UNet_128x128_bs16_t1000_e6100.ckpt"
)

epsSoftplusDDPMCondV3 = CFG2(
    epochs=10000,
    size=128,
    batch_size=16,
    timesteps=1000,
    parameterization="eps",
    condition_type="v3",
    schedule="softplus",
    model_name="epsSoftplusDDPMCondV3",
    model_path="./checkpoints/epsSoftplusDDPMCondV3/UNet_128x128_bs16_t1000_e4900.ckpt"
)

x0SoftplusDDPMCondV3 = CFG2(
    epochs=10000,
    size=128,
    batch_size=16,
    timesteps=1000,
    parameterization="x0",
    condition_type="v3",
    schedule="softplus",
    model_name="x0SoftplusDDPMCondV3",
    model_path="./checkpoints/x0SoftplusDDPMCondV3/UNet_128x128_bs16_t1000_e4700.ckpt"
)

x0SoftplusDDPM = CFG2(
    epochs=10000,
    size=128,
    batch_size=16,
    timesteps=1000,
    parameterization="x0",
    condition_type=None,
    schedule="softplus",
    model_name="x0SoftplusDDPM",
    model_path="./checkpoints/x0SoftplusDDPM/UNet_128x128_bs16_t1000_e1100.ckpt"
)

x0SoftplusDDPMCondV11 = CFG2(
    epochs=10000,
    size=128,
    batch_size=16,
    timesteps=1000,
    parameterization="x0",
    condition_type="v1",
    schedule="softplus",
    model_name="x0SoftplusDDPMCondV11",
    # model_path="./checkpoints/x0SoftplusDDPM/UNet_128x128_bs16_t1000_e1100.ckpt"
)

epsLinearDDPM = CFG2(
    epochs=10000,
    size=128,
    batch_size=16,
    timesteps=1000,
    parameterization="eps",
    condition_type=None,
    schedule="linear",
    model_name="epsLinearDDPM",
    # model_path="./checkpoints/x0SoftplusDDPM/UNet_128x128_bs16_t1000_e1100.ckpt"
)

epsCosineDDPM = CFG2(
    epochs=10000,
    size=128,
    batch_size=16,
    timesteps=1000,
    parameterization="eps",
    condition_type=None,
    schedule="cosine",
    model_name="epsCosineDDPM",
    # model_path="./checkpoints/x0SoftplusDDPM/UNet_128x128_bs16_t1000_e1100.ckpt"
)

epsSoftplusDDPM = CFG2(
    epochs=10000,
    size=128,
    batch_size=16,
    timesteps=1000,
    parameterization="eps",
    condition_type=None,
    schedule="softplus",
    model_name="epsSoftplusDDPM",
    # model_path="./checkpoints/x0SoftplusDDPM/UNet_128x128_bs16_t1000_e1100.ckpt"
)

epsSoftplusDDPMCondV3Latent= CFG2(
    epochs=10000,
    size=128,
    batch_size=16,
    timesteps=1000,
    parameterization="eps",
    condition_type="v3",
    schedule="softplus",
    model_name="epsSoftplusDDPMCondV3Latent",
    # model_path="./checkpoints/x0SoftplusDDPM/UNet_128x128_bs16_t1000_e1100.ckpt"
)

configs = [
    epsLinearDDPMCondV1,
    epsCosineDDPMCondV1,
    epsSoftplusDDPMCondV1,
    x0SoftplusDDPMCondV1,
    epsSoftplusDDPMCondV2,
    x0SoftplusDDPMCondV2,
    epsCosineDDPMCondV1_T,
    epsSoftplusDDPMCondV3,
    x0SoftplusDDPMCondV3,
    x0SoftplusDDPM,
    x0SoftplusDDPMCondV11,
    epsLinearDDPM,
    epsCosineDDPM,
    epsSoftplusDDPM,
    epsSoftplusDDPMCondV3Latent
]