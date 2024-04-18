
==================== // IDUN - GPU Recources // ====================

Jupyter Slurm Connection:
    ssh -L 8888:127.0.0.1:8888 -J kristiwk@idun-login1.hpc.ntnu.no kristiwk@idun-xx-xx

IDUN Modules:
    module load Python/3.10.4-GCCcore-11.3.0

Train Command Initiliazation:
    python main.py --epochs 1000 --size 128 --batch_size 16 --timesteps 1000 --parameterization eps --backbone True --save_model ConditionalV2

    python main.py --epochs 10000 --size 128 --batch_size 16 --timesteps 1000 --parameterization eps --model_path ./checkpoints/UNet_128x128_bs16_t1000_e400 --save_model ConditionalV2
    