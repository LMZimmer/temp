#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake
#SBATCH --array=1-2
#SBATCH -c 4
#SBATCH -t 01-00:00 # time (D-HH:MM)
#SBATCH -o logs/%A-%a.o
#SBATCH -e logs/%A-%a.e
#SBATCH --gres=gpu:0  # reserves no GPUs

# Info
echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"

# Setup
# Activate conda environment
source ~/.bashrc
conda activate nb_301

# Activate your conda/venv environment prior to executing this

# Increase number of parameter free-operations
if [ 1 -eq $SLURM_ARRAY_TASK_ID ]; then
  PYTHONPATH=$PWD python surrogate_models/analysis/op_list_parameter_free_op_increase.py --model_log_dir=$1 --nasbench_data=/home/siemsj/projects/nasbench_201_2/analysis/nb_301_v13
  exit $?
fi
