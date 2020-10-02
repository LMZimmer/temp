#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake
#SBATCH --array=1-50
#SBATCH -c 2
#SBATCH --mem 30000 # memory pool for all cores (8GB)
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
export PYTHONPATH=$PWD

# Arrayjob
python surrogate_models_hpo/run_bohb.py --array_id $SLURM_ARRAY_TASK_ID --total_num_workers 50 --num_iterations 1200 --run_id $SLURM_ARRAY_JOB_ID --working_directory=experiments/hpo_v12/gnn_gin_ranking_m_0.01 --model gnn_gin --data_root=projects/nasbench_201_2/analysis/nb_301_v112 --min_budget=128 --max_budget=128

# Done
echo "DONE"
echo "Finished at $(date)"
