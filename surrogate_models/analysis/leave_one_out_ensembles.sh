#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake
#SBATCH --array=9-9
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
export PYTHONPATH=$PWD

CURRENT=3


# Not COMBO
if [ 1 -eq $SLURM_ARRAY_TASK_ID ]; then
  PYTHONPATH=$PWD python surrogate_models/fit_model.py --model $1 --nasbench_data=/home/user/projects/nasbench_201_2/analysis/nb_301_v13 --data_config_path surrogate_models/configs/data_configs/leave_one_optimizer_out/not_combo.json --log_dir=experiments/surrogate_models/loo_ensemble/$1_$CURRENT/combo/$2 --seed $2 --data_splits_root surrogate_models/configs/data_splits/leave_out_combo
  exit $?
fi

# Not DE
if [ 2 -eq $SLURM_ARRAY_TASK_ID ]; then
  PYTHONPATH=$PWD python surrogate_models/fit_model.py --model $1 --nasbench_data=/home/user/projects/nasbench_201_2/analysis/nb_301_v13 --data_config_path surrogate_models/configs/data_configs/leave_one_optimizer_out/not_de.json --log_dir=experiments/surrogate_models/loo_ensemble/$1_$CURRENT/de/$2 --seed $2 --data_splits_root surrogate_models/configs/data_splits/leave_out_de
  exit $?
fi

# Not RE
if [ 3 -eq $SLURM_ARRAY_TASK_ID ]; then
  PYTHONPATH=$PWD python surrogate_models/fit_model.py --model $1 --nasbench_data=/home/user/projects/nasbench_201_2/analysis/nb_301_v13 --data_config_path surrogate_models/configs/data_configs/leave_one_optimizer_out/not_re.json --log_dir=experiments/surrogate_models/loo_ensemble/$1_$CURRENT/re/$2 --seed $2 --data_splits_root surrogate_models/configs/data_splits/leave_out_re
  exit $?
fi

# Not DARTS
if [ 4 -eq $SLURM_ARRAY_TASK_ID ]; then
  PYTHONPATH=$PWD python surrogate_models/fit_model.py --model $1 --nasbench_data=/home/user/projects/nasbench_201_2/analysis/nb_301_v13 --data_config_path surrogate_models/configs/data_configs/leave_one_optimizer_out/not_darts.json --log_dir=experiments/surrogate_models/loo_ensemble/$1_$CURRENT/darts/$2 --seed $2 --data_splits_root surrogate_models/configs/data_splits/leave_out_darts
  exit $?
fi

# Not TPE
if [ 5 -eq $SLURM_ARRAY_TASK_ID ]; then
  PYTHONPATH=$PWD python surrogate_models/fit_model.py --model $1 --nasbench_data=/home/user/projects/nasbench_201_2/analysis/nb_301_v13 --data_config_path surrogate_models/configs/data_configs/leave_one_optimizer_out/not_tpe.json --log_dir=experiments/surrogate_models/loo_ensemble/$1_$CURRENT/tpe/$2 --seed $2 --data_splits_root surrogate_models/configs/data_splits/leave_out_tpe
  exit $?
fi

# Not BANANAS
if [ 6 -eq $SLURM_ARRAY_TASK_ID ]; then
  PYTHONPATH=$PWD python surrogate_models/fit_model.py --model $1 --nasbench_data=/home/user/projects/nasbench_201_2/analysis/nb_301_v13 --data_config_path surrogate_models/configs/data_configs/leave_one_optimizer_out/not_bananas.json --log_dir=experiments/surrogate_models/loo_ensemble/$1_$CURRENT/bananas/$2 --seed $2 --data_splits_root surrogate_models/configs/data_splits/leave_out_bananas
  exit $?
fi

# Not GDAS
if [ 7 -eq $SLURM_ARRAY_TASK_ID ]; then
  PYTHONPATH=$PWD python surrogate_models/fit_model.py --model $1 --nasbench_data=/home/user/projects/nasbench_201_2/analysis/nb_301_v13 --data_config_path surrogate_models/configs/data_configs/leave_one_optimizer_out/not_gdas.json --log_dir=experiments/surrogate_models/loo_ensemble/$1_/gdas/$2 --seed $2 --data_splits_root surrogate_models/configs/data_splits/leave_out_gdas
  exit $?
fi

# Not PC DARTS
if [ 8 -eq $SLURM_ARRAY_TASK_ID ]; then
  PYTHONPATH=$PWD python surrogate_models/fit_model.py --model $1 --nasbench_data=/home/user/projects/nasbench_201_2/analysis/nb_301_v13 --data_config_path surrogate_models/configs/data_configs/leave_one_optimizer_out/not_pc_darts.json --log_dir=experiments/surrogate_models/loo_ensemble/$1_/pc_darts/$2 --seed $2 --data_splits_root surrogate_models/configs/data_splits/leave_out_pc_darts
  exit $?
fi

# Not one shot trajectories
if [ 9 -eq $SLURM_ARRAY_TASK_ID ]; then
  PYTHONPATH=$PWD python surrogate_models/fit_model.py --model $1 --nasbench_data=/home/user/projects/nasbench_201_2/analysis/nb_301_v13 --data_config_path surrogate_models/configs/data_configs/nb_301.json --log_dir=experiments/surrogate_models/loo_ensemble/$1_/skip_conn_splits/$2 --seed $2 --data_splits_root surrogate_models/configs/data_splits/skip_conn_splits
  exit $?
fi
