#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --constraint=cpu
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=8
#SBATCH --qos=regular
#SBATCH --account=mp107b
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=haridas@sas.upenn.edu

export OMP_NUM_THREADS=8
export OMP_PROC_BIND=master
export OMP_PLACES=sockets

srun -n 32 -c 8 --cpu-bind=sockets --distribution=block:block python make_source_map.py configs/make_source_map/nersc_atm.yaml
srun -n 32 -c 8 --cpu-bind=sockets --distribution=block:block python fit_source_map.py configs/make_source_map/nersc_atm.yaml
