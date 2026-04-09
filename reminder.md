# Commandes 

## Lancer une simulation
sbatch <lefichierbash>.sh

## Voir les logs de la simulation (ce que le terminal affiche)
cat log/resultat_<id_job>.out

## Voir les jobs en cours
squeue --me

## La conso après execution
 
sacct -j <jobid> --format=JobID,JobName,Elapsed,MaxRSS,MaxVMSize,AveCPU,State

## Les infos des partition du cluester
sinfo -o "%P %c %m %l"