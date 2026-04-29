#!/bin/bash
#SBATCH --job-name=21cm_1024
#SBATCH --output=log/result_%j.out
#SBATCH --error=log/err_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1  
#SBATCH --cpus-per-task=30
#SBATCH --time=01:00:00
#SBATCH --mem=100G
#SBATCH --partition=cpu_short


module load fftw/3.3.10/gcc-15.1.0-openmpi
source /gpfs/users/ouadjout/skadatachallenge/.venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH=/gpfs/users/ouadjout/skadatachallenge:$PYTHONPATH

python3 emulator/inference.py

