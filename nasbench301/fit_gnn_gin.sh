#!/bin/bash
#SBATCH -p partition_name
#SBATCH -c 4
#SBATCH -o logs/%A-%a.o
#SBATCH -e logs/%A-%a.e

# Info
echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"

# Setup
source ~/.bashrc
conda activate nb_301
export PYTHONPATH=$PWD

# Arrayjob
python surrogate_models/fit_model.py --model gnn_gin --nasbench_data projects/nasbench_201_2/analysis/nb_301_v13 --data_config_path surrogate_models/configs/data_configs/nb_301.json

# Done
echo "DONE"
echo "Finished at $(date)"
