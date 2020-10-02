import glob
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from analysis.global_nasbench_plots import compute_depth_of_cell_simple_path
from surrogate_models import utils
from surrogate_models.utils import ConfigLoader, evaluate_metrics

sns.set_style('whitegrid')

surrogate_model_colors = {
    'GIN': 'green',
    'LGB': 'blue',
    'XGB': 'orange'
}

surrogate_model_markers = {
    'GIN': 'x',
    'LGB': '^',
    'XGB': 'o'
}


def load_surrogate_model(model_log_dir):
    # Load config
    data_config = json.load(open(os.path.join(model_log_dir, 'data_config.json'), 'r'))
    model_config = json.load(open(os.path.join(model_log_dir, 'model_config.json'), 'r'))

    # Instantiate model
    surrogate_model = utils.model_dict[model_config['model']](data_root='None', log_dir=None,
                                                              seed=data_config['seed'], data_config=data_config,
                                                              model_config=model_config)

    # Load the model
    surrogate_model.load(os.path.join(model_log_dir, 'surrogate_model.model'))
    return surrogate_model


def analyze_cell_topology():
    config_loader = ConfigLoader('configspace.json')

    # Load groundtruth data
    cell_depths = [compute_depth_of_cell_simple_path(config_loader[gt][0])[0] for gt in glob.glob(
        '/home/user/projects/nasbench_201_2/analysis/nb_301_cell_topology/cell_topology_analysis/results_*.json')]

    gt_data_paths = [gt for gt in glob.glob(
        '/home/user/projects/nasbench_201_2/analysis/nb_301_cell_topology/cell_topology_analysis/results_*.json')]

    surrogate_models = {
        'GIN': load_surrogate_model(
            '/home/user/projects/nasbench_201_2/experiments/surrogate_models/gnn_gin/20200919-135631-6'),
        'LGB': load_surrogate_model(
            '/home/user/projects/nasbench_201_2/experiments/surrogate_models/lgb/20200919-135720-6'),
        'XGB': load_surrogate_model(
            '/home/user/projects/nasbench_201_2/experiments/surrogate_models/xgb/20200919-135720-6')
    }

    surrogate_model_results = {
        'cell_depth': cell_depths}

    for surrogate_model_name, surrogate_model in surrogate_models.items():
        test_metrics, preds, targets = surrogate_model.evaluate(gt_data_paths)
        surrogate_model_results[surrogate_model_name + '_preds'] = preds
        surrogate_model_results[surrogate_model_name + '_targets'] = targets

    fig, ax_left = plt.subplots(figsize=(4, 3))
    ax_left.set_ylabel('sKendall Tau')

    for surrogate_model in surrogate_models:
        idx = 0
        for cell_depth, group in pd.DataFrame(surrogate_model_results).groupby('cell_depth'):
            preds, targets = group['{}_preds'.format(surrogate_model)], group['{}_targets'.format(surrogate_model)]
            metrics = evaluate_metrics(targets, preds, prediction_is_first_arg=False)
            print(surrogate_model, cell_depth, metrics)
            ax_left.scatter(cell_depth, metrics['kendall_tau_1_dec'],
                            marker=surrogate_model_markers[surrogate_model.upper()],
                            label=surrogate_model.upper() if idx == 0 else None,
                            c=surrogate_model_colors[surrogate_model.upper()])
            idx += 1

    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    '''
    ax_right = ax_left.twinx()  # instantiate a second axes that shares the same x-axis
    ax_right.set_ylabel('R² (x)')

    for surrogate_model in surrogate_models:
        idx = 0
        for cell_depth, group in pd.DataFrame(surrogate_model_results).groupby('cell_depth'):
            preds, targets = group['{}_preds'.format(surrogate_model)], group['{}_targets'.format(surrogate_model)]
            metrics = evaluate_metrics(targets, preds, prediction_is_first_arg=False)
            ax_right.scatter(cell_depth, metrics['r2'], marker='x',
                             c=surrogate_model_colors[surrogate_model.upper()])
            idx += 1
    '''
    ax_left.set_xlabel('Cell Depth')
    plt.tight_layout()
    plt.savefig('surrogate_models/analysis/cell_topology_analysis.pdf')


if __name__ == "__main__":
    analyze_cell_topology()
