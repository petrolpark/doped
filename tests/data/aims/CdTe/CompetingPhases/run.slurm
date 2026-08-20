#!/bin/bash
#SBATCH -A kavanagh-sl2-cpu
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=76
#SBATCH --time=12:00:00
#SBATCH -J aims_CdTe_competing_phases
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

#! Build a list of competing-phase directories automatically: any folder
#! containing an aims_std subdirectory counts as a phase. This ignores
#! logs/ and anything else that isn't a real phase folder.
PHASES=()
for d in */aims_std; do
    PHASES+=("${d%/aims_std}")
done

PHASE=${PHASES[$SLURM_ARRAY_TASK_ID]}

if [ -z "$PHASE" ]; then
    echo "No phase at array index $SLURM_ARRAY_TASK_ID (found ${#PHASES[@]} phases total)"
    exit 1
fi

echo "Task $SLURM_ARRAY_TASK_ID handling phase: $PHASE"

cd "$SLURM_SUBMIT_DIR/$PHASE/aims_std"
pwd
mpirun -n $SLURM_NTASKS $AIMS_EXE > aims.out

echo "Finished phase: $PHASE"
