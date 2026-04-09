import sys
import py21cmfast as p21c
import os
print("Bonjour de la part du cluster !")
print(f"Version Python utilisée : {sys.version}")   
SLURM_CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
print(f"Utilisation de {SLURM_CPUS} threads via Slurm.")