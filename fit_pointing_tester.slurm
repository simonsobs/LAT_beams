#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=3:00:00
#SBATCH --constraint=cpu
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=8
#SBATCH --qos=interactive
#SBATCH --account=mp107b

export OMP_NUM_THREADS=8

srun -n 32 python fit_pointing.py configs/fit_pointing/mars_nersc.yaml -o --obs_ids obs_1744157073_lati5_101
