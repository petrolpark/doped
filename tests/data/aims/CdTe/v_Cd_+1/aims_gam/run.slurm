#!/bin/bash
#SBATCH -A kavanagh-sl2-cpu
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=76
#SBATCH --time=8:00:00
#SBATCH -J aims_test
#SBATCH --output=test_%j.out
#SBATCH --error=test_%j.err

module purge
module load rhel8/default-icl intel/compilers/2020.4 intel/mkl/2020.4 intel/impi/2020.4/intel
export OMP_NUM_THREADS=1
export I_MPI_PIN=1

cd ~/FHIaims/testcases/doped_tests/CdTe/v_Cd_+1/aims_gam
mpirun -n 76 ~/FHIaims/bin/aims > aims.out
