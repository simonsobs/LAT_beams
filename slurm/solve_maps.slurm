#!/bin/bash
#SBATCH --nodes=4
#SBATCH --time=20:00:00
#SBATCH --constraint=cpu
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=8
#SBATCH --qos=regular
#SBATCH --account=mp107b

# export OMP_PROC_BIND=spread
# export OMP_PLACES=threads
export OMP_NUM_THREADS=8

srun -n 64 --cpu-bind=none python solve_beam_map.py configs/make_source_map/nersc_atm.yaml
