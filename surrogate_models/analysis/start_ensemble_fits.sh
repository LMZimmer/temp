#!/bin/bash
SEED=1
LIMIT=10

until [ $SEED -gt $LIMIT ]
do
   # yes | sbatch surrogate_models/analysis/leave_one_out_ensembles.sh lgb $SEED

   yes | sbatch surrogate_models/analysis/leave_one_out_ensembles.sh gnn_gin $SEED

   yes | sbatch surrogate_models/analysis/leave_one_out_ensembles.sh xgb $SEED
   ((SEED++))
done
