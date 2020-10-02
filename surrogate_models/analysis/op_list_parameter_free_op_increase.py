import glob
import json
import os

import click
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams

from surrogate_models import utils

matplotlib.use('Agg')

rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'font.size': 14})

@click.command()
@click.option('--model_log_dir', type=click.STRING, help='Experiment directory')
@click.option('--nasbench_data', type=click.STRING, help='Path to nasbench root directory')
def op_list_skip_connect_increase(model_log_dir, nasbench_data):
    # Load config
    data_config = json.load(open(os.path.join(model_log_dir, 'data_config.json'), 'r'))
    model_config = json.load(open(os.path.join(model_log_dir, 'model_config.json'), 'r'))

    # Instantiate model
    surrogate_model = utils.model_dict[model_config['model']](data_root=nasbench_data, log_dir=None,
                                                              seed=data_config['seed'], data_config=data_config,
                                                              model_config=model_config)

    # Load the model from checkpoint
    surrogate_model.load(os.path.join(model_log_dir, 'surrogate_model.model'))

    # Set the same seed as used during the training.
    np.random.seed(data_config['seed'])

    flatten = lambda l: [item for sublist in l for item in sublist]
    test_paths = flatten(
        [json.load(open(val_opt)) for val_opt in glob.glob(os.path.join(model_log_dir, '*_test_paths.json'))])

    # Take all datapoints in a diagonal fidelity and transform them to the other fidelities.
    # Iterate through the parameter types
    ratio_skip_connection_in_cell_dict = {'max_pool_3x3': {}, 'avg_pool_3x3': {}, 'skip_connect': {}}
    for parameter_free_op in ratio_skip_connection_in_cell_dict.keys():
        surrogate_model.config_loader.parameter_free_op_increase_type = parameter_free_op
        # Progressively increase the ratio of the selected parameter free operation
        for ratio_parameter_free_op_in_cell in np.arange(0, 1.1, 1 / 8):
            surrogate_model.config_loader.ratio_parameter_free_op_in_cell = ratio_parameter_free_op_in_cell
            val_pred_results = []
            for i in range(4):
                _, val_preds, _ = surrogate_model.evaluate(test_paths)
                val_pred_results.extend(1 - val_preds / 100)
            ratio_skip_connection_in_cell_dict[parameter_free_op][ratio_parameter_free_op_in_cell] = val_pred_results

    num_op_type_rep = len(ratio_skip_connection_in_cell_dict['max_pool_3x3'][0]) * 9
    data_dict = {
        "Num. operations": np.array(
            [*flatten([i * 8] * len(ratio_skip_connection_in_cell_dict['max_pool_3x3'][0]) for i in
                      np.arange(0, 1.1, 1 / 8))] * 3, dtype=np.int),
        "Operation type": ["Num. avg. pool"] * num_op_type_rep + ["Num. max pool"] * num_op_type_rep +
                          ["Num. skip conn."] * num_op_type_rep,
        "Validation error": [*flatten(list(ratio_skip_connection_in_cell_dict["max_pool_3x3"].values())),
                             *flatten(list(ratio_skip_connection_in_cell_dict["avg_pool_3x3"].values())),
                             *flatten(list(ratio_skip_connection_in_cell_dict["skip_connect"].values()))]}
    df = pd.DataFrame(data_dict)
    gt_data= {
        op_type: [json.load(open(gt)) for gt in glob.glob(
            os.path.join(nasbench_data, 'only_{}'.format(op_type), '*'))]
        for op_type in ratio_skip_connection_in_cell_dict.keys()}

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.violinplot(x="Num. operations", y="Validation error", hue="Operation type", data=df, linewidth=0.1, cut=0)
    gt_matcher_dict = {
        'max_pool_3x3': 'orange', 'avg_pool_3x3': 'blue', 'skip_connect': 'green'
    }

    for op_type, evals in gt_data.items():
        val_errors = [1 - eval['info'][0]['val_accuracy_final'] / 100 for eval in evals]
        plt.scatter(np.ones_like(val_errors) * 8, val_errors, s=1, alpha=0.2, c=gt_matcher_dict[op_type], label=None)
    ax.set_yscale("log")
    ax.set_ylim([0.045, 0.25])
    ax.set_yticks([0.05, 0.06, 0.1, 0.2])
    ax.set_yticklabels(["0.05", "", "0.1", "0.2"])
    plt.grid(True, which="both", ls="-", axis="y", alpha=0.5)
    plt.legend(loc="upper left", prop={'size': 12})
    # adjust_box_widths(fig, 0.9)
    plt.savefig(os.path.join(model_log_dir, 'parameter_free_op_ratio.pdf'))
    plt.close()


if __name__ == "__main__":
    op_list_skip_connect_increase()
