#!/bin/bash

#SBATCH --account=rrg-wperciva
#SBATCH --job-name=NUTS
#SBATCH --output=NUTS.out
#SBATCH --time=7-00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --mail-user=a2crespi@uwaterloo.ca
#SBATCH --mail-type=BEGIN,END,FAIL

module load julia/1.10.0

srun julia --project=/home/acrespi/scratch/Flinch -t 32 /home/acrespi/scratch/Flinch/src/NUTS_sampler.jl
