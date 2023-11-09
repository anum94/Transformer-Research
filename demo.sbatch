#!/bin/bash
#SBATCH -N 1
#SBATCH -p lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH -o lrz/runs/legacy/dump.out
#SBATCH -e lrz/runs/legacy/dump.err
#SBATCH --time=800

srun --container-image ~/demo.sqsh torchrun  ~/transformer-research/run_hf_eval.py