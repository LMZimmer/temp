import glob
import json
import os

import click
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy import stats

from surrogate_models import utils
from surrogate_models.utils import evaluate_metrics

matplotlib.use('Agg')


@click.command()
@click.option('--analysis_directory', type=click.STRING, help='Directory of the leave one optimizer out analysis')
@click.option('--nasbench_data', type=click.STRING, help='Path to nasbench root directory')
def leave_one_out_analysis(analysis_directory, nasbench_data):
    result_dict = {}
    data_min, data_max = 100, 0
    optimizers = ['darts', 'bananas', 'combo', 'de', 're', 'tpe', 'pc_darts', 'gdas', 'drnas']
    result_loader = utils.ResultLoader('', '', '', 0)
    for optimizer in optimizers:
        try:
            model_log_dir = glob.glob(os.path.join(analysis_directory, optimizer, '*', '*'))[0]
            data_config = json.load(open(os.path.join(model_log_dir, 'data_config.json'), 'r'))
            model_config = json.load(open(os.path.join(model_log_dir, 'model_config.json'), 'r'))

            # Instantiate model
            surrogate_model = utils.model_dict[model_config['model']](data_root=nasbench_data, log_dir=None,
                                                                      seed=data_config['seed'], data_config=data_config,
                                                                      model_config=model_config)
            surrogate_model.load(os.path.join(model_log_dir, 'surrogate_model.model'))

            left_out_optimizer_paths = result_loader.filter_duplicate_dirs(
                glob.glob(os.path.join(nasbench_data, optimizer, '*')))
            _, val_preds, val_true = surrogate_model.evaluate(left_out_optimizer_paths)
            data_min = min(data_min, *val_preds, *val_true)
            data_max = max(data_max, *val_preds, *val_true)
            if type(val_preds) is not list:
                val_preds, val_true = val_preds.tolist(), val_true.tolist()
            result_dict[optimizer] = {'val_preds': val_preds, 'val_true': val_true}
        except FileNotFoundError as e:
            pass

    fig = plt.figure(figsize=(3, 20))
    grid = ImageGrid(fig, 111,  # as in plt.subplot(111)
                     nrows_ncols=(len(optimizers), 1),
                     axes_pad=0.5,
                     share_all=True)
    counter = 0
    statistics = {}
    for ax, (optimizer, results) in zip(grid, result_dict.items()):
        if counter == 0:
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            counter += 1
        ax.scatter(results['val_preds'], results['val_true'], s=1, alpha=0.15)
        kendall = stats.kendalltau(np.round(np.array(results['val_preds']), decimals=1), results['val_true'])

        statistics[optimizer] = {
            **evaluate_metrics(results['val_true'], results['val_preds'], prediction_is_first_arg=False)
        }
        ax.set_xlim(data_min, data_max)
        ax.set_ylim(data_min, data_max)
        ax.plot([data_min, data_max], [data_min, data_max], c='r', alpha=0.3)
        ax.grid(True, which="both", ls="-", alpha=0.1)
        ax.set_title('{} - K:{}'.format(optimizer.upper(), "{:.3f}".format(kendall.correlation)))

    plt.savefig(os.path.join(analysis_directory, 'loo_analysis.png'), dpi=600)

    # Dump statistics
    json.dump(statistics, open(os.path.join(analysis_directory, 'statistics.json'), 'w'))

    # Dump the results from the analysis
    json.dump(result_dict, open(os.path.join(analysis_directory, 'analysis_results.json'), 'w'))


if __name__ == "__main__":
    leave_one_out_analysis()
