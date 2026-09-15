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

# OMP_NUM_THREADS=2
# NUM_FUTURES=8
# OMP_PLACES=cores
# OMP_PROC_BIND=close
# srun -n 16 -c 16 --cpu-bind=cores --distribution=block:block python make_stacked_map.py configs/make_source_map/nersc_atm.yaml

OMP_NUM_THREADS=16
srun -n 8 -c 16 --cpu-bind=cores --distribution=block:block python fit_stacked_map.py configs/make_source_map/nersc_atm.yaml
