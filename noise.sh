#!/bin/bash
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --job-name=noise
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --mem=32GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpossaz@eafit.edu.co

source ~/.bashrc

conda activate img2img

python -m img2img.utils.noise_information \
    --embed_path assets/projected/dragon.pt \
    --seed_a 42 \
    --seed_b 43 \
    --mixing_strategy "dither" \
    --percent 0.0