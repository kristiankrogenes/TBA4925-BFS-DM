#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=40G
#SBATCH --job-name="Rtba4925kwk"
#SBATCH --output=./scripts/RjupyterV100.out
#SBATCH --gres=gpu:v100:1
#SBATCH --constraint=gpu32g

module load Anaconda3/2022.10
python main.py --config_name epsOrtoConditionalSoftplus