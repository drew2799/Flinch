#!/bin/bash

#SBATCH --account=rrg-wperciva
#SBATCH --job-name=PATHinit
#SBATCH --output=PATHinit.out
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=4000G
#SBATCH --mail-user=a2crespi@uwaterloo.ca
#SBATCH --mail-type=BEGIN,END,FAIL

module load julia/1.10.0

srun julia --project=/home/acrespi/scratch/Flinch -t 32 /home/acrespi/scratch/Flinch/src/PATHinit.jl
