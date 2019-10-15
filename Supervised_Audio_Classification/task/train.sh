#!/bin/bash
# created: Sep 21, 2019 21:03 AM
# author: xiehuang
#SBATCH -J task_train
#SBATCH -o task_train.out_%j
#SBATCH -e task_train.err_%j
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:k80:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -t 12:00:00
#SBATCH --mem-per-cpu=12000
#SBATCH --mail-type=END
#SBATCH --mail-user=huang.xie@tuni.fi

# note, this job requests a total of 1 cores and 1 GPGPU cards
# note, submit the job from taito-gpu.csc.fi
# commands to manage the batch script
#   submission command
#     sbatch [script-file]
#   status command
#     squeue -u xiehuang
#   termination command
#     scancel [jobid]

# For more information
#   man sbatch
#   more examples in Taito GPU guide in
#   http://research.csc.fi/taito-gpu

module purge
module load gcc
module load cuda/10.0
module list

# example run commands
srun python3 /wrk/xiehuang/DONOTREMOVE/FSD2018/task/main.py

# This script will print some usage statistics to the
# end of file: task_train.out
# Use that to improve your resource request estimate
# on later jobs.
seff $SLURM_JOBID
