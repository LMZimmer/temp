import json
import os
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
from IPython import embed

from surrogate_models import utils
from surrogate_models.ensemble import Ensemble


def compute_no_weight_operations(config_space_instance):
    """
    Compute the number of no-weight operations of a cell
    :param config_space_instance:
    :return:
    """
    config_dict = config_space_instance.get_dictionary()
    no_weight_ops = {'normal': {'max_pool_3x3': 0,
                                'avg_pool_3x3': 0,
                                'skip_connect': 0},
                     'reduce': {'max_pool_3x3': 0,
                                'avg_pool_3x3': 0,
                                'skip_connect': 0}}
    for cell_type in ['normal', 'reduce']:
        for edge in range(14):
            op = config_dict.get(
                'NetworkSelectorDatasetInfo:darts:edge_{}_{}'.format(cell_type, edge), None)
            if op in no_weight_ops[cell_type]:
                no_weight_ops[cell_type][op] += 1
    return no_weight_ops


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


def compute_kl_divergence(p, q):
    """Compute the KL divergence KL(p | q) from two lists assuming gaussian distributions"""
    m1 = np.mean(p)
    m2 = np.mean(q)
    s1 = np.std(p)
    s2 = np.std(q)

    kl_div = np.log(s2 / s1) + (s1 ** 2 + (m1 - m2) ** 2) / (2 * s2 ** 2) - 1 / 2
    return kl_div


# Run: python3 surrogate_models/analysis/ensemble_noise_analysis.py --nasbench_data /home/user/projects/nasbench_201_2/analysis/nb_301_v13 --ensemble_parent_dir /home/user/projects/nasbench_201_2/experiments/surrogate_models/ensembles/gnn_gin_3
@click.command()
@click.option('--nasbench_data', type=click.STRING, help='path to nasbench root directory')
@click.option('--ensemble_parent_dir', type=click.STRING, help='Directory containing the ensemble members')
def ensemble_noise_analysis(nasbench_data, ensemble_parent_dir):
    # Read data
    config_performances = dict()

    results_paths = [filename for filename in Path(nasbench_data).rglob('*.json')]
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
                "train_acc": train_accuracy,
                "val_acc": val_accuracy,
                "test_acc": test_accuracy,
                "num_parameters": num_parameters,
                "config_path": config_path,
                "config": config
            }

            if config_hash not in config_performances.keys():
                config_performances[config_hash] = []
            config_performances[config_hash].append(info_dict)

        except Exception as e:
            print('Exception', e, config_path)

    # Delete configs with few evaluations
    min_num_evals = 5
    config_hashes = list(config_performances.keys())
    for chash in config_hashes:
        if not len(config_performances[chash]) > (min_num_evals - 1):
            del config_performances[chash]
    print("==> Found %i configs with more than %i evals" % (len(config_performances), min_num_evals))

    # Get model directories
    member_dirs = get_ensemble_member_dirs(ensemble_parent_dir)
    member_dirs_dict = group_dirs_by_surrogate_type(member_dirs)
    print("==> Found ensemble member directories:", member_dirs_dict)

    # Load an ensemble for each surrogate type
    surrogate_ensemble = None
    for surrogate_type, member_dirs in member_dirs_dict.items():

        # Load config
        print("==> Loading %s configs..." % surrogate_type)
        model_log_dir = member_dirs[0]
        data_config = json.load(open(os.path.join(model_log_dir, 'data_config.json'), 'r'))
        model_config = json.load(open(os.path.join(model_log_dir, 'model_config.json'), 'r'))

        # Load ensemble
        print("==> Loading %s ensemble..." % surrogate_type)
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

    # Set the same seed as used during the training.
    np.random.seed(data_config['seed'])

    # Evaluate ensemble
    print("==> Evaluating ensemble...")
    config_statistics = dict()
    for chash, evaluation_list in config_performances.items():

        groundtruth_peformances = []
        surrogate_predictions = []

        for ind, info_dict in enumerate(evaluation_list):
            config = info_dict["config"]
            surrogate_prediction = surrogate_ensemble.query(config)/100
            info_dict["ensemble_prediction"] = surrogate_prediction

            surrogate_predictions.append(surrogate_prediction)
            groundtruth_peformances.append(info_dict["val_acc"]/100)

        stddevs_gt = [np.std(groundtruth_peformances[0:ind]) for ind in range(len(groundtruth_peformances))]
        stddevs_pred = [np.std(surrogate_predictions[0:ind]) for ind in range(len(surrogate_predictions))]

        config_statistics[chash] = {}
        config_statistics[chash]["val_acc_gt"] = np.mean(groundtruth_peformances[1:])
        config_statistics[chash]["single_seed_pred"] = groundtruth_peformances[0]
        config_statistics[chash]["surrogate_prediction"] = np.mean(surrogate_predictions)
        config_statistics[chash]["val_acc_stddevs_groundtruth"] = stddevs_gt
        config_statistics[chash]["val_acc_stddevs_pred"] = stddevs_pred
        config_statistics[chash]["kl_div"] = compute_kl_divergence(p=groundtruth_peformances, q=surrogate_predictions)

    print("==> Ensemble evaluation finished.")

    # Mean stddevs
    stddevs_gt = [config_statistics[chash]["val_acc_stddevs_groundtruth"][-1] for chash in config_statistics.keys()]
    stddevs_pred = [config_statistics[chash]["val_acc_stddevs_pred"][-1] for chash in config_statistics.keys()]
    mean_stddev_gt = np.mean(stddevs_gt)
    mean_stddev_pred = np.mean(stddevs_pred)
    stddev_mae = mean_absolute_error(stddevs_gt, stddevs_pred)
    print("==> Groundtruth stddev vs. predicted stddev: %f / %f" % (mean_stddev_gt, mean_stddev_pred))
    print("==> MAE between groundtruth stddev and predicted stddev: %f" % stddev_mae)

    # MAE from 2 to 5/10 seeds
    # NOTE: Done for the groundtruth in analysis/global_nasbench_plots
    gts = [config_statistics[chash]["val_acc_gt"] for chash in config_statistics.keys()]
    seed_preds = [config_statistics[chash]["single_seed_pred"] for chash in config_statistics.keys()]
    surr_preds = [config_statistics[chash]["surrogate_prediction"] for chash in config_statistics.keys()]

    seed_mae = mean_absolute_error(gts, seed_preds)
    surrogate_mae = mean_absolute_error(gts, surr_preds)

    print("==> 1 seed pred MAE: %f" %seed_mae)
    print("==> surrogate pred MAE: %f" %surrogate_mae)


    # KL divergences
    kl_divs = [config_statistics[chash]["kl_div"] for chash in config_statistics.keys()]
    mean_kl_div = np.mean(kl_divs)
    print("==> Mean KL divergence: %f" % mean_kl_div)

    embed()

    # Log
    noise_results = {
        "Mean stddev groundtruth": mean_stddev_gt,
        "Mean stddev predicted": mean_stddev_pred,
        "Predicted vs gt stddev MAE": stddev_mae,
        "Mean KL divergence": mean_kl_div,
        "1_seed_pred_MAE": seed_mae,
        "Surrogate pred MAE": surrogate_mae
    }

    for key, val in noise_results.items():
        if isinstance(val, dict):
            for subkey, subval in val.items():
                noise_results[key][subkey] = np.float64(subval)
        else:
            try:
                noise_results[key] = np.float64(val)
            except:
                pass

    json.dump(noise_results, open(os.path.join(ensemble_parent_dir, "noise_statistics.json"), "w"))

    """
    # Plots for ensemble variance on all data
    # Stddevs on edge casesa
    # lowpar_dirs = list(Path(nasbench_data).rglob('only_*'))
    # low_par_result_paths = []
    # for lowpar_dir in lowpar_dirs:
    #    low_par_result_paths += list(Path(lowpar_dir).rglob('*.json'))
    configs_low_par = []
    val_acc_low_par_gt = []
    val_acc_low_par_preds = []
    val_acc_stddev = []
    num_parameters_low_par = []
    no_weight_ops_list = []

    for config_path in tqdm(results_paths, desc='Reading dataset'):
        try:
            config_space_instance, val_accuracy, test_accuracy, json_file = config_loader[config_path]
            config = config_space_instance.get_dictionary()
            no_weight_ops = compute_no_weight_operations(config_space_instance)

            num_parameters = json_file['info'][0]['model_parameters']
            train_accuracy = json_file['info'][0]['train_accuracy_final']

            val_accuracy = val_accuracy / 100
            configs_low_par.append(config)
            val_acc_low_par_gt.append(val_accuracy)
            num_parameters_low_par.append(num_parameters)
            no_weight_ops_list.append(no_weight_ops)

            preds = []
            for ind in range(10):
                pred = surrogate_ensemble.query(config)
                pred = pred / 100
                preds.append(pred)
            val_acc_low_par_preds.append(preds)
            val_acc_stddev.append(np.std(preds))

        except Exception as e:
            print('Exception', e, config_path)

    # NUM SKIP CONN VS VAL STDDEV (NORMAL)
    num_avg_pool = [cell["normal"]["avg_pool_3x3"] for cell in no_weight_ops_list]
    num_max_pool = [cell["normal"]["max_pool_3x3"] for cell in no_weight_ops_list]
    num_skip_conn = [cell["normal"]["skip_connect"] for cell in no_weight_ops_list]

    data_dict = {"Num operations": num_avg_pool + num_max_pool + num_skip_conn,
                 "Operation type": ["Num avg pool"] * len(num_avg_pool) + ["Num max pool"] * len(num_max_pool) + [
                     "Num skip conn."] * len(num_skip_conn),
                 "Ensemble variance": val_acc_stddev * 3}

    df = pd.DataFrame(data=data_dict)

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.violinplot(x="Num operations", y="Ensemble variance", hue="Operation type", data=df, linewidth=0.1, cut=0)

    # ax.set_yscale("log")
    # ax.set_ylim([0.045, 0.25])
    # ax.set_yticks([0.05, 0.06, 0.1, 0.2])
    # ax.set_yticklabels(["0.05", "", "0.1", "0.2"])
    plt.grid(True, which="both", ls="-", axis="y", alpha=0.5)
    plt.legend(loc="upper right", prop={'size': 12})
    plt.savefig(
        os.path.join(ensemble_parent_dir, 'num_operations_vs_ensemble_variance_normal_%s.png' % model_config['model']),
        dpi=300, alpha=0.5)
    plt.close()

    # NUM SKIP CONN VS VAL STDDEV (REDUCTION)
    num_avg_pool = [cell["reduce"]["avg_pool_3x3"] for cell in no_weight_ops_list]
    num_max_pool = [cell["reduce"]["max_pool_3x3"] for cell in no_weight_ops_list]
    num_skip_conn = [cell["reduce"]["skip_connect"] for cell in no_weight_ops_list]

    data_dict = {"Num operations": num_avg_pool + num_max_pool + num_skip_conn,
                 "Operation type": ["Num avg pool"] * len(num_avg_pool) + ["Num max pool"] * len(num_max_pool) + [
                     "Num skip conn."] * len(num_skip_conn),
                 "Ensemble variance": val_acc_stddev * 3}

    df = pd.DataFrame(data=data_dict)

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.violinplot(x="Num operations", y="Ensemble variance", hue="Operation type", data=df, linewidth=0.1, cut=0)

    # ax.set_yscale("log")
    # ax.set_ylim([0.045, 0.25])
    # ax.set_yticks([0.05, 0.06, 0.1, 0.2])
    # ax.set_yticklabels(["0.05", "", "0.1", "0.2"])
    plt.grid(True, which="both", ls="-", axis="y", alpha=0.5)
    plt.legend(loc="upper right", prop={'size': 12})
    plt.savefig(os.path.join(ensemble_parent_dir,
                             'num_operations_vs_ensemble_variance_reduction_%s.png' % model_config['model']), dpi=300)
    plt.close()
    """

if __name__ == "__main__":
    ensemble_noise_analysis()
