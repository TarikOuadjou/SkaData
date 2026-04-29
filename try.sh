#!/bin/bash
#SBATCH --job-name=test_loader
#SBATCH --output=log/resultat_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --mem=8G

mkdir -p log

module load fftw/3.3.10/gcc-15.1.0-openmpi
source /gpfs/users/ouadjout/skadatachallenge/.venv/bin/activate

cd /gpfs/users/ouadjout/skadatachallenge  # ← ton répertoire racine du projet
python3 main.py