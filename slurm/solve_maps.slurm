#!/bin/bash
#SBATCH --nodes=2
#SBATCH --time=10:00:00
#SBATCH --constraint=cpu
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=8
#SBATCH --qos=regular
#SBATCH --account=mp107b

# export OMP_PROC_BIND=spread
# export OMP_PLACES=threads
export OMP_NUM_THREADS=4

srun -n 32 --cpu-bind=none python solve_beam_map.py configs/make_source_map/nersc_atm.yaml
