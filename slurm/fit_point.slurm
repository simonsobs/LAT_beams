#!/bin/bash
#SBATCH --nodes=2
#SBATCH --time=20:00:00
#SBATCH --constraint=cpu
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=8
#SBATCH --qos=regular
#SBATCH --account=mp107b

export OMP_NUM_THREADS=4
export OMP_PROC_BIND=master
export OMP_PLACES=sockets

srun -n 128 -c 4 --cpu-bind=sockets --distribution=block:block python fit_pointing.py configs/fit_pointing/nersc.yaml -f 16 -o
