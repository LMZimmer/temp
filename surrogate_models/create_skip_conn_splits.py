import glob
import json
import os
import shutil
import time

import click
import matplotlib

from surrogate_models import utils
from nas_benchmark.plots_iclr.one_shot import *

from IPython import embed

matplotlib.use('Agg')


def replace_string_booleans(config_dict):
    for key in config_dict:
        if config_dict[key]=="False":
            config_dict[key]=False
        if config_dict[key]=="True":
            config_dict[key]=True
    return config_dict


@click.command()
def create_data_splits():

    # Get all paths
    default_data_splits_path = "surrogate_models/configs/data_splits/default_split"
    train_paths = json.load(open(os.path.join(default_data_splits_path, "train_paths.json"), "r"))
    val_paths = json.load(open(os.path.join(default_data_splits_path, "val_paths.json"), "r"))
    test_paths = json.load(open(os.path.join(default_data_splits_path, "test_paths.json"), "r"))

    # Get train trajectory configs
    configs, _  = load_groundtruth_trajectories(darts_indices_min=1601, darts_indices_max=1951)
    train_configs = [replace_string_booleans(configs["DARTS"][traj][ind]) for traj in range(len(configs["DARTS"])) for ind in range(len(configs["DARTS"][traj]))]

    # Get test trajectory configs
    configs, _  = load_groundtruth_trajectories(darts_indices_min=1951, darts_indices_max=2201)
    test_configs = [config for method in configs.keys() for config in configs[method]]
    test_configs = [replace_string_booleans(test_configs[traj][ind]) for traj in range(len(test_configs)) for ind in range(len(test_configs[traj]))]

    train_test_configs = train_configs + test_configs

    # Resplit
    train_paths = train_paths + test_paths
    train_paths_cleaned = []
    val_paths_cleaned = []

    trajectory_test_paths  = []

    for path in train_paths:
        config = json.load(open(path, "r"))["optimized_hyperparamater_config"]
        if config not in test_configs:
            train_paths_cleaned.append(path)
        else:
            trajectory_test_paths.append(path)
    
    for path in val_paths:
        config = json.load(open(path, "r"))["optimized_hyperparamater_config"]
        if config not in test_configs:
            val_paths_cleaned.append(path)
        else:
            trajectory_test_paths.append(path)

    test_paths_cleaned = trajectory_test_paths

    print("==> Train/val/test %i/%i/%i" %(len(train_paths_cleaned), len(val_paths_cleaned), len(test_paths_cleaned)))
    print("==> Removed %i configs" %(len(train_paths)+len(val_paths) - len(train_paths_cleaned)-len(val_paths_cleaned)))

    # Save data splits
    print("==> Saving data splits")
    splits_log_dir = "surrogate_models/configs/data_splits/skip_conn_splits/"
    os.makedirs(splits_log_dir, exist_ok=True)
    json.dump(train_paths_cleaned, open(os.path.join(splits_log_dir, "train_paths.json"), "w"))
    json.dump(val_paths_cleaned, open(os.path.join(splits_log_dir, "val_paths.json"), "w"))
    json.dump(test_paths_cleaned, open(os.path.join(splits_log_dir, "test_paths.json"), "w"))


if __name__ == "__main__":
    create_data_splits()
