#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=40G
#SBATCH --job-name="BFSDM_Latent"
#SBATCH --output=./scripts/BFSDM_Latent.out
#SBATCH --gres=gpu:1

module load Python/3.10.4-GCCcore-11.3.0
python main.py --config_name epsSoftplusDDPMCondV3Latent