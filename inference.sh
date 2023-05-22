#!/bin/bash
#SBATCH --partition=accel-2
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=5:00:00
#SBATCH --job-name=inference_img2img
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --mem=32GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpossaz@eafit.edu.co

echo "Starting job"

source ~/.bashrc

conda activate img2img

python -m img2img.utils.inference \
    data_files/downloaded/laionmaxaesth \
    assets/configs/base.json \
    runs/19343/model_4000.pt \
    --seed 835 \
    --generated_per_input 4 \
    --batch_size 1 \
    --offset 0 \
    --total_inputs 480 \
    --output_path inference_one_base