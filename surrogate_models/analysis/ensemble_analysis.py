import json
import os
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np

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


def compare(list1, list2):
    """Checks wether the elements in two lists are the same. Also considers duplicates."""
    return sorted(list1) == sorted(list2)


def get_test_configs(ensemble_member_logdir):
    """Loads the paths, configs, val scores, test scores from the log directory of a surrogate model (= one ensemble member)"""

    test_paths = [filename for filename in Path(ensemble_member_logdir).rglob('*test_paths.json')]

    test_results = []
    for test_path in test_paths:
        test_results.append(read_json(test_path))
    test_results = flatten_list(test_results)

    configs = []
    val_scores = []
    test_scores = []
    for test_result in test_results:
        results = read_json(test_result)
        configs.append(results["optimized_hyperparamater_config"])
        val_scores.append(results["info"][0]["val_accuracy_final"])
        test_scores.append(results["test_accuracy"])

    return test_results, configs, val_scores, test_scores


def check_test_splits(surrogate_ensemble):
    """Performs checks on the data paths:
    
    i) Checks if the logged test paths are the same (*_test.json in the training directory)

    ii) Checks if the attribute .test_paths is the same for each member of the ensemble

    iii) Checks if .test_paths are the same as the logged test paths
    """

    # Get ensemble member directories
    ensemble_member_dirs = surrogate_ensemble.member_logdirs

    # Collect config hashes from all test results from all members
    member_test_config_hashes = []
    for member_dir in ensemble_member_dirs:
        _, configs, _, _ = get_test_configs(member_dir)
        config_hashes = [hash(config.__repr__()) for config in configs]
        member_test_config_hashes.append(config_hashes)

    all_hashes = flatten_list(member_test_config_hashes)

    # Check uniqueness of configs
    configs_of_member = [len(config_hashes) for config_hashes in member_test_config_hashes]
    num_unique_configs_of_member = [len(np.unique(config_hashes)) for config_hashes in member_test_config_hashes]
    num_total_unique_configs = len(np.unique(all_hashes))

    print("==> Found", configs_of_member, "test configs for individual ensemble members.")
    print("==> Found", num_unique_configs_of_member, "unique test configs for individual ensemble members.")
    print("==> Found %i unique test configs between all ensemble members." % num_total_unique_configs)

    if len(np.unique(num_unique_configs_of_member)) > 1 or num_total_unique_configs != num_unique_configs_of_member[0]:
        raise ValueError("Ensemble members have different test configs!")

    # Check .test_paths attribute
    previous_test_paths_attribute = surrogate_ensemble.ensemble_members[0].test_paths
    for ensemble_member, ensemble_member_dir in zip(surrogate_ensemble.ensemble_members, ensemble_member_dirs):
        test_paths_attribute = ensemble_member.test_paths
        # Check if it is the same between members
        same_attributes = compare(test_paths_attribute, previous_test_paths_attribute)
        # Check if they are the same as the logged paths
        result_paths, _, _, _ = get_test_configs(ensemble_member_dir)
        same_attribute_and_logged = compare(test_paths_attribute, result_paths)

        if not same_attributes:
            # print("==> Ensemble members have different test path attributes!")
            raise ValueError("Ensemble members have different test path attributes!")
        if not same_attribute_and_logged:
            # print("==> Ensemble members test path attribute has different paths than the logged test paths!")
            raise ValueError("Ensemble members test path attribute has different paths than the logged test paths!")

        previous_test_paths_attribute = test_paths_attribute


# Run: python3 surrogate_models/analysis/ensemble_analysis.py --nasbench_data /home/user/projects/nasbench_201_2/analysis/nb_301_v10 --ensemble_parent_dir /home/user/nasbench_201_2/experiments/surrogate_models/ensembles/lgb --data_splits_root /home/user/nasbench_201_2/surrogate_models/configs/data_splits/default_split
@click.command()
@click.option('--nasbench_data', type=click.STRING, help='path to nasbench root directory')
@click.option('--ensemble_parent_dir', type=click.STRING, help='Directory containing the ensemble members')
@click.option('--data_splits_root', type=click.STRING, help='')
def ensemble_analysis(nasbench_data, ensemble_parent_dir, data_splits_root):
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
        train_paths = json.load(open(os.path.join(data_splits_root, 'train_paths.json'), 'r'))
        val_paths = json.load(open(os.path.join(data_splits_root, 'val_paths.json'), 'r'))
        test_paths = json.load(open(os.path.join(data_splits_root, 'test_paths.json'), 'r'))

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

        surrogate_ensemble_single.load(model_paths=member_dirs,
                                       train_paths=train_paths,
                                       val_paths=val_paths,
                                       test_paths=test_paths)

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
    print("==> Evaluating ensemble performance...")
    train_metrics, train_preds, train_stddevs, train_targets = surrogate_ensemble.evaluate_ensemble(train_paths,
                                                                                                    apply_noise=False)
    val_metrics, val_preds, val_stddevs, val_targets = surrogate_ensemble.validate_ensemble(apply_noise=False)
    test_metrics, test_preds, test_stddevs, test_targets = surrogate_ensemble.test_ensemble(apply_noise=False)
    print('==> Ensemble train metrics', train_metrics)
    print('==> Ensemble val metrics', val_metrics)
    print('==> Ensemble test metrics', test_metrics)

    train_metrics_with_noise, train_preds_with_noise, _, _ = surrogate_ensemble.evaluate_ensemble(train_paths,
                                                                                                  apply_noise=True)
    val_metrics_with_noise, val_preds_with_noise, _, _ = surrogate_ensemble.validate_ensemble(apply_noise=True)
    test_metrics_with_noise, test_preds_with_noise, _, _ = surrogate_ensemble.test_ensemble(apply_noise=True)
    print('==> Ensemble train metrics (noisy)', train_metrics_with_noise)
    print('==> Ensemble val metrics (noisy)', val_metrics_with_noise)
    print('==> Ensemble test metrics (noisy)', test_metrics_with_noise)

    train_mean_stddev = np.mean(train_stddevs)
    val_mean_stddev = np.mean(val_stddevs)
    test_mean_stddev = np.mean(test_stddevs)
    print("==> Mean ensemble stddev on train set %f" % train_mean_stddev)
    print("==> Mean ensemble stddev on validation set %f" % val_mean_stddev)
    print("==> Mean ensemble stddev on test set %f" % test_mean_stddev)

    # Plots
    fig_val = utils.scatter_plot(np.array(train_preds), np.array(train_targets), xlabel='Predicted', ylabel='True',
                                 title='')
    fig_val.savefig(os.path.join(ensemble_parent_dir, 'pred_vs_true_train.jpg'))
    plt.close()

    fig_val = utils.scatter_plot(np.array(val_preds), np.array(val_targets), xlabel='Predicted', ylabel='True',
                                 title='')
    fig_val.savefig(os.path.join(ensemble_parent_dir, 'pred_vs_true_val.jpg'))
    plt.close()

    fig_test = utils.scatter_plot(np.array(test_preds), np.array(test_targets), xlabel='Predicted', ylabel='True',
                                  title='')
    fig_test.savefig(os.path.join(ensemble_parent_dir, 'pred_vs_true_test.jpg'))
    plt.close()

    fig_val = utils.scatter_plot(np.array(train_preds_with_noise), np.array(train_targets), xlabel='Predicted',
                                 ylabel='True', title='')
    fig_val.savefig(os.path.join(ensemble_parent_dir, 'pred_vs_true_train_noisy.jpg'))
    plt.close()

    fig_val_noise = utils.scatter_plot(np.array(val_preds_with_noise), np.array(val_targets), xlabel='Predicted',
                                       ylabel='True', title='')
    fig_val_noise.savefig(os.path.join(ensemble_parent_dir, 'pred_vs_true_val_noisy.jpg'))
    plt.close()

    fig_test_noise = utils.scatter_plot(np.array(test_preds_with_noise), np.array(test_targets), xlabel='Predicted',
                                        ylabel='True', title='')
    fig_test_noise.savefig(os.path.join(ensemble_parent_dir, 'pred_vs_true_test_noisy.jpg'))
    plt.close()

    # Test query_mean method
    # print("==> Testing ensemble predictions with query mean...")
    # test_results, configs, val_scores, test_scores = get_test_configs(member_dirs[0])
    # preds = [surrogate_ensemble.query_mean(config) for config in configs]
    # metrics = utils.evaluate_metrics(val_scores, preds, prediction_is_first_arg=False)
    # print("==> Metrics on test data (query mean):", metrics)

    # Perform cell topology performance analysis
    print("==> Checking configs from cell topology analysis...")
    test_paths = [filename for filename in Path(
        "/home/user/projects/nasbench_201_2/analysis/nb_301_cell_topology/cell_topology_analysis").rglob('*.json')]
    test_metrics_topology, ensemble_predictions_topology, stddevs_topology, targets_topology = surrogate_ensemble.evaluate_ensemble(
        test_paths, apply_noise=False)
    print("==> Metrics on cell topology analysis:", test_metrics_topology)

    # Log
    results = {
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "train_metrics_with_noise": train_metrics_with_noise,
        "val_metrics_with_noise": val_metrics_with_noise,
        "test_metrics_with_noise": test_metrics_with_noise,
        "train_mean_stddev": train_mean_stddev,
        "val_mean_stddev": val_mean_stddev,
        "test_mean_stddev": test_mean_stddev,
        "test_results_cell_topology": test_metrics_topology}

    for key, val in results.items():
        if isinstance(val, dict):
            for subkey, subval in val.items():
                results[key][subkey] = np.float64(subval)
        else:
            try:
                results[key] = np.float64(val)
            except:
                pass

    json.dump(results, open(os.path.join(ensemble_parent_dir, "ensemble_performance.json"), "w"))

    # Check wether test splits are the same for ensemble members and logged paths
    # print("==> Checking test splits of ensemble members...")
    # check_test_splits(surrogate_ensemble)
    # print("==> All checks passed.")


if __name__ == "__main__":
    ensemble_analysis()
