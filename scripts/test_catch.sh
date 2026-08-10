#!/bin/bash
#SBATCH --job-name=MVD_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_b300_sxm6_ac_2g.67gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=5-12:00:00
#SBATCH --output=outputs/logs/test_dino_MV.log

nohup bash -c '
run_name="test_catch"

python src/eval.py \
    --dataset-name private_echo \
    --n-classes 4 \
    --log-dir "outputs/${run_name}" \
    --use-attention \
    --start-ckpt "outputs/train_catch/ckpt_ep_1.pth"

' > "outputs/logs/eval.log" 2>&1 &