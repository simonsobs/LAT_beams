#!/bin/bash
#SBATCH --nodes=2
#SBATCH --time=20:00:00
#SBATCH --constraint=cpu
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=8
#SBATCH --qos=regular
#SBATCH --account=mp107b

export OMP_NUM_THREADS=4

srun -n 64 python fit_pointing.py configs/fit_pointing/neptune_nersc.yaml -f 8 -r -o
srun -n 64 python fit_pointing.py configs/fit_pointing/uranus_nersc.yaml -f 8 -r -o
srun -n 64 python fit_pointing.py configs/fit_pointing/mars_nersc.yaml -f 8 -r -o
srun -n 64 python fit_pointing.py configs/fit_pointing/saturn_nersc.yaml -f 8 -r -o
