#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake
#SBATCH -c 4
#SBATCH -t 01-00:00 # time (D-HH:MM)
#SBATCH -o logs/%A-%a.o
#SBATCH -e logs/%A-%a.e

# Info
echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"

# Setup
# Activate conda environment
source ~/.bashrc
conda activate nb_301

# Activate your conda/venv environment prior to executing this
PYTHONPATH=$PWD python surrogate_models/analysis/ensemble_noise_analysis.py --nasbench_data /home/user/projects/nasbench_201_2/analysis/nb_301_v13 --ensemble_parent_dir /home/user/projects/nasbench_201_2/experiments/surrogate_models/ensembles/gnn_gin_3

PYTHONPATH=$PWD python surrogate_models/analysis/ensemble_noise_analysis.py --nasbench_data /home/user/projects/nasbench_201_2/analysis/nb_301_v13 --ensemble_parent_dir /home/user/projects/nasbench_201_2/experiments/surrogate_models/ensembles/xgb_2

PYTHONPATH=$PWD python surrogate_models/analysis/ensemble_noise_analysis.py --nasbench_data /home/user/projects/nasbench_201_2/analysis/nb_301_v13 --ensemble_parent_dir /home/user/projects/nasbench_201_2/experiments/surrogate_models/ensembles/lgb_2
