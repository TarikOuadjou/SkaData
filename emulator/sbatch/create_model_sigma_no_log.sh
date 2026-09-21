#!/bin/bash
#SBATCH --job-name=21cm_1024
#SBATCH --output=emulator/sigma_model_no_log/log/result_%j.out
#SBATCH --error=emulator/sigma_model_no_log/log/err_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1  
#SBATCH --cpus-per-task=30
#SBATCH --time=03:00:00
#SBATCH --mem=40G
#SBATCH --partition=cpu_med


module load fftw/3.3.10/gcc-15.1.0-openmpi
source /gpfs/users/ouadjout/skadatachallenge/.venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH=/gpfs/users/ouadjout/skadatachallenge:$PYTHONPATH

python3 emulator/sigma_model_no_log/get_distrib.py

