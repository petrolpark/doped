#!/bin/bash
#SBATCH -A kavanagh-sl2-cpu
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=76
#SBATCH --time=12:00:00
#SBATCH -J aims_defects
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#! --array is set on the command line when you submit (see instructions)

module purge
module load rhel8/default-icl intel/compilers/2020.4 intel/mkl/2020.4 intel/impi/2020.4/intel
export OMP_NUM_THREADS=1
export I_MPI_PIN=1

#! EDIT THIS to the actual path of your FHI-aims executable:
AIMS_EXE=~/FHIaims/bin/aims

cd $SLURM_SUBMIT_DIR

#! Build a list of defect directories automatically: any folder containing
#! an aims_gam subdirectory counts as a defect. This ignores logs/ and
#! anything else that isn't a real defect folder.
DEFECTS=()
for d in */aims_gam; do
    DEFECTS+=("${d%/aims_gam}")
done

DEFECT=${DEFECTS[$SLURM_ARRAY_TASK_ID]}

if [ -z "$DEFECT" ]; then
    echo "No defect at array index $SLURM_ARRAY_TASK_ID (found ${#DEFECTS[@]} defects total)"
    exit 1
fi

echo "Task $SLURM_ARRAY_TASK_ID handling defect: $DEFECT"

cd "$SLURM_SUBMIT_DIR/$DEFECT/aims_gam"
pwd
mpirun -n $SLURM_NTASKS $AIMS_EXE > aims.out

echo "Finished defect: $DEFECT"
