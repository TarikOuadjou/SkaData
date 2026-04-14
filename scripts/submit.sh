#!/bin/bash
#SBATCH --job-name=21cm_1024
#SBATCH --output=log/result_%j_%a.out
#SBATCH --error=log/err_%j_%a.err
#SBATCH --array=0-127%10        # 1024 points, max 50 en parallèle 
#SBATCH --nodes=1
#SBATCH --ntasks=1  
#SBATCH --cpus-per-task=10
#SBATCH --time=01:00:00
#SBATCH --mem=40G
#SBATCH --partition=cpu_med


mkdir -p log results

module load fftw/3.3.10/gcc-15.1.0-openmpi
source /gpfs/users/ouadjout/skadatachallenge/.venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH=/gpfs/users/ouadjout/skadatachallenge:$PYTHONPATH

python3 scripts/run_one.py $SLURM_ARRAY_TASK_ID

rm -rf /gpfs/workdir/ouadjout/cache/job_$(printf "%04d" $SLURM_ARRAY_TASK_ID)