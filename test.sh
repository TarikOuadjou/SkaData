#!/bin/bash
#SBATCH --job-name=mon_premier_job   # Nom de votre job
#SBATCH --output=resultat_%j.out     # Fichier de sortie (%j = ID du job)
#SBATCH --nodes=1                  # 1 seul nœud (machine)
#SBATCH --ntasks=1                   # Nombre de tâches
#SBATCH --cpus-per-task=30         # Réserve 30 cœurs pour le multi-threading
#SBATCH --time=00:05:00              # Temps maximum (HH:MM:SS)
#SBATCH --mem=4G                     # Mémoire vive allouée
module load fftw/3.3.10/gcc-15.1.0-openmpi
source /gpfs/users/ouadjout/skadatachallenge/.venv/bin/activate

python3 test.py