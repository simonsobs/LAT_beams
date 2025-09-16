#!/bin/bash
if [[ ! -z "${SLURM_MEM_PER_CPU}" ]]; then
unset SLURM_MEM_PER_CPU
unset SLURM_OPEN_MODE
fi

export OMP_NUM_THREADS=8

module use /global/common/software/sobs/perlmutter/modulefiles
module load soconda/stable
module load cudatoolkit

cat << EOF > fit_point_nightly.slurm 
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=1:30:00
#SBATCH --constraint=cpu
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=8
#SBATCH --qos=regular
#SBATCH --account=mp107b

srun -n 32 python fit_pointing.py configs/fit_pointing/mars.yaml -l 36
EOF

sbatch fit_point_nightly.slurm
