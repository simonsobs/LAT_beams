#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --constraint=cpu
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=4
#SBATCH --qos=regular
#SBATCH --account=mp107b
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=haridas@sas.upenn.edu

export OMP_NUM_THREADS=4
export OMP_PROC_BIND=master
export OMP_PLACES=sockets

srun -n 64 -c 4 --cpu-bind=sockets --distribution=block:block python fit_pointing.py configs/fit_pointing/nersc.yaml -f 16
