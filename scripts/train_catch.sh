#!/bin/bash
#SBATCH --job-name=MVD_dice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_b300_sxm6_ac_2g.67gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=5-12:00:00
#SBATCH --output=outputs/logs/train_dino_MV.log


run_name="train_catch"

python src/train.py \
    --epochs 10 \
    --dataset-name private_echo \
    --n-classes 4 \
    --batch-size 8 \
    --log-dir "outputs/${run_name}" \
    --use-attention \
    --run-name "${run_name}"


