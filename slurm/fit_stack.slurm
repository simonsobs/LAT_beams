#!/bin/bash
#SBATCH --qos=shared
#SBATCH --constraint=cpu
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --account=mp107b
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=haridas@sas.upenn.edu

export OMP_NUM_THREADS=32
python fit_stacked_map.py configs/make_source_map/nersc_atm.yaml
