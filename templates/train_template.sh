#!/bin/bash
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --job-name=text_embedding
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --mem=32GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user={{ email }}

echo "Starting job"

source ~/.bashrc

conda activate img2img

if [ -z "$SLURM_JOB_ID" ]; then
    export SLURM_JOB_ID=$(python -c "import random; print(random.randint(0, 1000000000000000))")
fi

python -m img2img.training \
    --run_name $SLURM_JOB_ID \
    --data_path data_files/downloaded/laion6p0 \
    --model_config_path {{ config_path }} \
    --seed 42 \
    --data_split 0.8 \
    --gradient_accumulation_steps 1 \
    --batch_size 32 \
    --training_steps 1000 \
    --validation_frequency 100 \
    --checkpoint_frequency 250 \
    --benchmarks_path assets/benchmarks/ \
    --wandb_project img2img_reprojector_train \
    --img_extension ".jpg" \
    --upload_checkpoints_to_wandb True \