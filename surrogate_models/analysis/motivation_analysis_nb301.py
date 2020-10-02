import random
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from IPython import embed

import os
import json
import click
import logging
import numpy as np
import pandas as pd
import collections
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error
from surrogate_models import utils
from surrogate_models.ensemble import Ensemble


def get_ensemble_member_dirs(parent_dir):
    """Get the directories of potential ensemble members within a directory (recursively)"""
    member_dirs = [os.path.dirname(filename) for filename in Path(parent_dir).rglob('*surrogate_model.model')]
    return member_dirs

def group_dirs_by_surrogate_type(surrogate_member_dirs):
    """Create a dictionary that groups directories by the type of surrogate they contain by reading the model config"""

    member_dirs_grouped_by_surr_type = dict()

    for surrogate_member_dir in surrogate_member_dirs:
        model_config_dir = os.path.join(surrogate_member_dir, "model_config.json")
        model_config = read_json(model_config_dir)
        surrogate_type = model_config["model"]
        if surrogate_type not in member_dirs_grouped_by_surr_type.keys():
            member_dirs_grouped_by_surr_type[surrogate_type] = []
        member_dirs_grouped_by_surr_type[surrogate_type].append(surrogate_member_dir)
    return member_dirs_grouped_by_surr_type

def read_json(p):
    """Read a file via json.load"""
    with open(str(p), "r") as f: 
        data = json.load(f) 
    return data

def flatten_list(nested_list):
    """Flattes a list of lists"""
    flat_list = [entry for sublist in nested_list for entry in sublist]
    return flat_list


# Run: python3 surrogate_models/analysis/motivation_analysis_nb301.py --apt_data /home/user/logs/de_seed/ --ensemble_parent_dir /home/user/nasbench_201_2/experiments/surrogate_models/ensembles/lgb
@click.command()
@click.option('--nasbench_data', type=click.STRING, help='path to nasbench root directory', default='None')
@click.option('--apt_data', type=click.STRING, help='path to apt log dir', default='None')
@click.option('--ensemble_parent_dir', type=click.STRING, help='Directory containing the ensemble members')
def motivation_analysis(nasbench_data, ensemble_parent_dir, apt_data):
 
    # Read data
    config_performances = dict()

    if nasbench_data!="None":
        multi_run_dir = os.path.join(nasbench_data, "de_multi_seed")
        results_paths = [filename for filename in Path(multi_run_dir).rglob('*.json')]
        config_loader = utils.ConfigLoader('configspace.json')
        for config_path in tqdm(results_paths, desc='Reading dataset'):
        # for config_path in tqdm(results_paths[0:5000], desc='Reading dataset'):
            try:
                config_space_instance, val_accuracy, test_accuracy, json_file = config_loader[config_path]
                config = config_space_instance.get_dictionary()
            
                num_parameters = json_file['info'][0]['model_parameters']
                train_accuracy = json_file['info'][0]['train_accuracy_final']

                config_hash = hash(config.__repr__())
                info_dict = {
                        "train_acc" : train_accuracy,
                        "val_acc" : val_accuracy,
                        "test_acc" : test_accuracy,
                        "num_parameters" : num_parameters,
                        "config_path" : config_path,
                        "config" : config
                        }
            
                if config_hash not in config_performances.keys():
                    config_performances[config_hash] = []
                config_performances[config_hash].append(info_dict)

            except Exception as e:
                print('Exception', e, config_path)

    else:
        results_paths = [filename for filename in Path(apt_data).rglob('*final_output.json')]
        for config_path in tqdm(results_paths, desc='Reading dataset'):

            results = read_json(str(config_path))

            config = results["optimized_hyperparamater_config"]
            train_accuracy = results["info"][0]["train_accuracy"]
            val_accuracy = results["info"][0]["val_accuracy"]
            test_accuracy = results["test_accuracy"]
            num_parameters = results["info"][0]["model_parameters"]
            
            config_hash = hash(config.__repr__())
            info_dict = {
                    "train_acc" : train_accuracy,
                    "val_acc" : val_accuracy,
                    "test_acc" : test_accuracy,
                    "num_parameters" : num_parameters,
                    "config_path" : config_path,
                    "config" : config
                    }

            if config_hash not in config_performances.keys():
                config_performances[config_hash] = []
            config_performances[config_hash].append(info_dict)

    # Delete configs with few evaluations
    min_num_evals = 5
    config_hashes = list(config_performances.keys())
    for chash in config_hashes:
        if not len(config_performances[chash])>(min_num_evals-1):
            del config_performances[chash]
    print("==> Found %i configs with more than %i evals" %(len(config_performances), min_num_evals))

    # Get model directories
    member_dirs = get_ensemble_member_dirs(ensemble_parent_dir)
    member_dirs_dict = group_dirs_by_surrogate_type(member_dirs)
    print("==> Found ensemble member directories:", member_dirs_dict)

    # Load an ensemble for each surrogate type
    surrogate_ensemble = None
    for surrogate_type, member_dirs in member_dirs_dict.items():

        # Load config
        print("==> Loading %s configs..." %surrogate_type)
        model_log_dir = member_dirs[0]
        data_config = json.load(open(os.path.join(model_log_dir, 'data_config.json'), 'r'))
        model_config = json.load(open(os.path.join(model_log_dir, 'model_config.json'), 'r'))

        # Load ensemble
        print("==> Loading %s ensemble..." %surrogate_type)
        surrogate_ensemble_single = Ensemble(member_model_name=model_config['model'],
                                             data_root='None', 
                                             log_dir=ensemble_parent_dir, 
                                             starting_seed=data_config['seed'],
                                             model_config=model_config, 
                                             data_config=data_config, 
                                             ensemble_size=len(member_dirs),
                                             init_ensemble=False)

        surrogate_ensemble_single.load(model_paths=member_dirs)

        # Combine different model types
        if surrogate_ensemble is None:
            surrogate_ensemble = surrogate_ensemble_single
        else:
            for member_model in surrogate_ensemble_single.ensemble_members:
                surrogate_ensemble.add_member(member_model)

    print("==> Ensemble creation completed.")

    surrogate_model = surrogate_ensemble.ensemble_members[0]

    # Set the same seed as used during the training.
    np.random.seed(data_config['seed'])

    # Collect means
    val_acc_single = []
    val_acc_mean = []
    val_acc_pred = []
    val_acc_pred_ens = []

    print("==> Infering...")

    for chash, info_dict_list in config_performances.items():
        config = info_dict_list[0]["config"]
        result_dirs = [str(info_dict["config_path"]) for info_dict in info_dict_list]
        seed_1_dir = np.where(["/1/" in result_path for result_path in result_dirs])[0]
        not_seed_1_dir = np.where(["/1/" not in result_path for result_path in result_dirs])[0]
        
        config_val_accs = np.array([info_dict["val_acc"] for info_dict in info_dict_list])
        val_accs_without_seed_1 = config_val_accs[not_seed_1_dir]
        val_acc_seed_1 = config_val_accs[seed_1_dir][0]

        val_acc_mean.append(np.mean(val_accs_without_seed_1))
        val_acc_single.append(val_acc_seed_1)
        val_acc_pred.append(surrogate_model.query(config))
        val_acc_pred_ens.append(surrogate_ensemble.query(config))

    # Compare performances
    tabular_error = mean_absolute_error(val_acc_single, val_acc_mean)
    surrogate_error = mean_absolute_error(val_acc_pred, val_acc_mean)
    surrogate_ens_error = mean_absolute_error(val_acc_pred_ens, val_acc_mean)

    print("==> Surrogate model: %s" %model_config["model"])
    print("==> Tabular error: %f" %tabular_error)
    print("==> Surrogate error: %f" %surrogate_error)
    print("==> Surrogate ensemble error: %f" %surrogate_ens_error)

    results_dict = {
            "surrogate_model": model_config["model"],
            "tabular_error": np.float64(tabular_error),
            "surrogate_erorr": np.float64(surrogate_error),
            "surrogate_ensemble_error": np.float64(surrogate_ens_error)
            }

    outdir = os.path.join(ensemble_parent_dir, "motivation_experiment_results.json")
    json.dump(results_dict, open(outdir, "w"))


if __name__ == "__main__":
    motivation_analysis()
