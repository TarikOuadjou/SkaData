#!/bin/bash
#SBATCH --job-name=21cm_sobol
#SBATCH --output=log/resultat_%a.out
#SBATCH --error=log/error_%a.err
#SBATCH --array=0-3                  # 4 points Sobol → indices 0,1,2,3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --mem=40G

mkdir -p log results

module load fftw/3.3.10/gcc-15.1.0-openmpi
source /gpfs/users/ouadjout/skadatachallenge/.venv/bin/activate

python3 run_model.py $SLURM_ARRAY_TASK_ID