#!/bin/bash
#SBATCH --job-name=21cm_1024
#SBATCH --output=low_generate_data/log/result_%j_%a.out
#SBATCH --error=low_generate_data/log/err_%j_%a.err
#SBATCH --array=0-999%500      # 4096 points, max 10 en parallèle 
#SBATCH --nodes=1
#SBATCH --ntasks=1  
#SBATCH --cpus-per-task=3
#SBATCH --time=01:00:00
#SBATCH --mem=10G
#SBATCH --partition=cpu_med

REAL_ID=$((SLURM_ARRAY_TASK_ID + 11000))

mkdir -p low_generate_data/log results

module load fftw/3.3.10/gcc-15.1.0-openmpi
source /gpfs/users/ouadjout/skadatachallenge/.venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH=/gpfs/users/ouadjout/skadatachallenge:$PYTHONPATH

python3 low_generate_data/run_one.py $REAL_ID

rm -rf /gpfs/workdir/ouadjout/cache/job_$(printf "%04d" $REAL_ID)