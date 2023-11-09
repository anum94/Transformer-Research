#!/bin/bash
#SBATCH -N 1
#SBATCH -p lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH -o lrz/dump.out
#SBATCH -e lrz/dump.err
#SBATCH --time=800

srun --container-image ~/demo.sqsh deepspeed --num_gpus=1 --no_local_rank ~/transformer-research/main.py --config ~/transformer-research/summ.json --dsconfig ~/transformer-research/dsconfig_zero3.json