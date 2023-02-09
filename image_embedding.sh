#!/bin/bash
#SBATCH --partition=accel-2
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --job-name=text_embedding
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --mem=32GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpossaz@eafit.edu.co

source ~/.bashrc

conda activate img2img

python -m img2img.data.preprocessing \
    --force --input_path data_files/downloaded/laion6p0 \
    --output_path data_files/processed/laion6p0 \
    --is_image \
    --batch_size 8 \
    --task_group_size 64 \
    --n_workers 2 \
    --n_thread_per_worker 4 