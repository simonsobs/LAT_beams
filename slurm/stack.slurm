#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=6:00:00
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

srun -n 32 -c 4 --cpu-bind=sockets --distribution=block:block python make_stacked_map.py configs/make_source_map/nersc_atm.yaml
export OMP_NUM_THREADS=32
python fit_stacked_map.py configs/make_source_map/nersc_atm.yaml
