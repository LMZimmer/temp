#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake
#SBATCH --array=1-9
#SBATCH -c 3
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

CURRENT=$(date +%s)

# Not COMBO
if [ 1 -eq $SLURM_ARRAY_TASK_ID ]; then
  PYTHONPATH=$PWD python surrogate_models/fit_model.py --model $1 --nasbench_data=/home/user/projects/nasbench_201_2/analysis/nb_301_v13 --data_config_path surrogate_models/configs/data_configs/leave_one_optimizer_out/not_combo.json --log_dir=experiments/surrogate_models/loo/$1_$CURRENT/combo
  exit $?
fi

# Not DE
if [ 2 -eq $SLURM_ARRAY_TASK_ID ]; then
  PYTHONPATH=$PWD python surrogate_models/fit_model.py --model $1 --nasbench_data=/home/user/projects/nasbench_201_2/analysis/nb_301_v13 --data_config_path surrogate_models/configs/data_configs/leave_one_optimizer_out/not_de.json --log_dir=experiments/surrogate_models/loo/$1_$CURRENT/de
  exit $?
fi

# Not RE
if [ 3 -eq $SLURM_ARRAY_TASK_ID ]; then
  PYTHONPATH=$PWD python surrogate_models/fit_model.py --model $1 --nasbench_data=/home/user/projects/nasbench_201_2/analysis/nb_301_v13 --data_config_path surrogate_models/configs/data_configs/leave_one_optimizer_out/not_re.json --log_dir=experiments/surrogate_models/loo/$1_$CURRENT/re
  exit $?
fi

# Not DARTS
if [ 4 -eq $SLURM_ARRAY_TASK_ID ]; then
  PYTHONPATH=$PWD python surrogate_models/fit_model.py --model $1 --nasbench_data=/home/user/projects/nasbench_201_2/analysis/nb_301_v13 --data_config_path surrogate_models/configs/data_configs/leave_one_optimizer_out/not_darts.json --log_dir=experiments/surrogate_models/loo/$1_$CURRENT/darts
  exit $?
fi

# Not TPE
if [ 5 -eq $SLURM_ARRAY_TASK_ID ]; then
  PYTHONPATH=$PWD python surrogate_models/fit_model.py --model $1 --nasbench_data=/home/user/projects/nasbench_201_2/analysis/nb_301_v13 --data_config_path surrogate_models/configs/data_configs/leave_one_optimizer_out/not_tpe.json --log_dir=experiments/surrogate_models/loo/$1_$CURRENT/tpe
  exit $?
fi

# Not BANANAS
if [ 6 -eq $SLURM_ARRAY_TASK_ID ]; then
  PYTHONPATH=$PWD python surrogate_models/fit_model.py --model $1 --nasbench_data=/home/user/projects/nasbench_201_2/analysis/nb_301_v13 --data_config_path surrogate_models/configs/data_configs/leave_one_optimizer_out/not_bananas.json --log_dir=experiments/surrogate_models/loo/$1_$CURRENT/bananas
  exit $?
fi

# Not GDAS
if [ 7 -eq $SLURM_ARRAY_TASK_ID ]; then
  PYTHONPATH=$PWD python surrogate_models/fit_model.py --model $1 --nasbench_data=/home/user/projects/nasbench_201_2/analysis/nb_301_v13 --data_config_path surrogate_models/configs/data_configs/leave_one_optimizer_out/not_gdas.json --log_dir=experiments/surrogate_models/loo/$1_$CURRENT/gdas
  exit $?
fi

# Not PC-DARTS
if [ 8 -eq $SLURM_ARRAY_TASK_ID ]; then
  PYTHONPATH=$PWD python surrogate_models/fit_model.py --model $1 --nasbench_data=/home/user/projects/nasbench_201_2/analysis/nb_301_v13 --data_config_path surrogate_models/configs/data_configs/leave_one_optimizer_out/not_pc_darts.json --log_dir=experiments/surrogate_models/loo/$1_$CURRENT/pc_darts
  exit $?
fi

# Not DRNAS
if [ 9 -eq $SLURM_ARRAY_TASK_ID ]; then
  PYTHONPATH=$PWD python surrogate_models/fit_model.py --model $1 --nasbench_data=/home/user/projects/nasbench_201_2/analysis/nb_301_v13 --data_config_path surrogate_models/configs/data_configs/leave_one_optimizer_out/not_drnas.json --log_dir=experiments/surrogate_models/loo/$1_$CURRENT/drnas
  exit $?
fi
