#!/bin/bash
#SBATCH --job-name=21cm_1024
#SBATCH --output=emulator/mdn_model/log/result_%j.out
#SBATCH --error=emulator/mdn_model/log/err_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1  
#SBATCH --cpus-per-task=30
#SBATCH --time=02:00:00
#SBATCH --mem=40G
#SBATCH --partition=cpu_med


module load fftw/3.3.10/gcc-15.1.0-openmpi
source /gpfs/users/ouadjout/skadatachallenge/.venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH=/gpfs/users/ouadjout/skadatachallenge:$PYTHONPATH

python3 emulator/mdn_model/train_model.py

