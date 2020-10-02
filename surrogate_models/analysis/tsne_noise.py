import random
from datetime import datetime
from pathlib import Path

import os
import json
import click
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.manifold import TSNE
from tqdm import tqdm

from analysis.global_nasbench_plots import compute_depth_of_cell_simple_path, compute_no_weight_operations
from surrogate_models.utils import ConfigLoader
from surrogate_models.ensemble import Ensemble

sns.set_style('whitegrid')

rcParams.update({'figure.autolayout': True})


@click.command()
@click.option('--nasbench_data', type=click.STRING, help='path to nasbench root directory')
@click.option('--model_log_dir', type=click.STRING, help='Experiment directory (ensemble)')
def plotting(nasbench_data, model_log_dir):
    # Load config
    print("==> Loading configs")
    data_config = json.load(open(os.path.join(model_log_dir, 'data_config.json'), 'r'))
    model_config = json.load(open(os.path.join(model_log_dir, 'model_config.json'), 'r'))

    # Instantiate model
    print("==> Loading model")
    surrogate_model = Ensemble(member_model_name=model_config['model'],
                               data_root=nasbench_data, 
                               log_dir=model_log_dir, 
                               starting_seed=data_config['seed'],
                               model_config=model_config, 
                               data_config=data_config, 
                               ensemble_size=5)

    surrogate_model.load(os.path.join(model_log_dir, 'surrogate_model.model'))

    # Set the same seed as used during the training.
    np.random.seed(data_config['seed'])

    current_time = datetime.now()
    config_loader = ConfigLoader('configspace.json')
    results_paths = [filename for filename in Path(nasbench_data).rglob('*.json')]  # + \
    config_vecs = []
    config_dicts = []
    val_errors = []
    optimizer_name = []
    normal_cell_depths = []
    reduce_cell_depths = []
    num_parameters = []
    num_skip_connect_normal = []
    predicted_noise = []
    print("==> Loading NB301 data")
    for config_path in tqdm(results_paths, desc='Reading dataset'):
        if "surrogate_model" in str(config_path):
            continue
        try:
            config_space_instance, val_accuracy, test_accuracy, json_file = config_loader[config_path]
            optimizer_name.append(config_path.parent.stem)
            config_vec = config_space_instance.get_array()
            config_vecs.append(config_vec)
            config_dicts.append(config_space_instance)
            normal_depth, reduce_depth = compute_depth_of_cell_simple_path(config_space_instance)
            normal_cell_depths.append(normal_depth)
            reduce_cell_depths.append(reduce_depth)
            num_skip_connect_normal.append(
                compute_no_weight_operations(config_space_instance)['normal']['skip_connect'])
            num_parameter = json_file['info'][0]['model_parameters']
            num_parameters.append(num_parameter)
            val_accuracy = json_file['info'][0]['val_accuracy']
            val_errors.append(1 - val_accuracy / 100)
            pred_stddev = surrogate_model.query_stddev(config_space_instance)
            predicted_noise.append(pred_stddev)
        except Exception as e:
            print('error', e, config_path)

    arch_ranking = ss.rankdata(predicted_noise)
    val_errors_group_by = pd.DataFrame(
        {'optimizer': optimizer_name,
         'val_error': val_errors,
         'cell_depth': normal_cell_depths}).groupby('optimizer')

    cell_depth_optimizer = pd.DataFrame(
        {'optimizer': [*optimizer_name, *optimizer_name],
         'val_error': [*val_errors, *val_errors],
         'Cell Depth': [*normal_cell_depths, *reduce_cell_depths],
         'Cell Type': [*['normal' for _ in normal_cell_depths],
                       *['reduce' for _ in reduce_cell_depths]]})

    print('########## STARTING TSNE')
    X_no_nan = np.nan_to_num(np.array(config_vecs), nan=0.0)
    X_emb = TSNE(n_components=2).fit_transform(X_no_nan)
    X_emb_group_by = np.split(X_emb, np.cumsum(np.unique(optimizer_name, return_counts=True)[1])[:-1])
    arch_ranking_group_by = pd.DataFrame(
        {'optimizer': optimizer_name, 'arch_ranking': predicted_noise, 'X_emb_0': X_emb[:, 0],
         'X_emb_1': X_emb[:, 1]}).groupby('optimizer')

    y_min, y_max, x_min, x_max = np.min(X_emb[:, 1]), np.max(X_emb[:, 1]), np.min(X_emb[:, 0]), np.max(X_emb[:, 0])

    # NORMAL AND REDUCTION CELL DEPTH COMPARISON
    print('########## CREATING FIGURES')
    plt.figure(figsize=(4, 3))
    # plt.title('NAS-Bench-201 {}, num. models: {}'.format(current_time.strftime("%b %d %Y %H:%M"), len(test_errors)))
    plt.xlabel('t-SNE 1st component')
    plt.ylabel('t-SNE 2nd component')
    ax = plt.gca()
    plt.scatter(X_emb[:, 0], X_emb[:, 1], s=2, alpha=0.15, c=predicted_noise, #norm=matplotlib.colors.LogNorm(),
                cmap=plt.get_cmap('viridis'))
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Predicted noise', rotation=270, labelpad=15)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(model_log_dir, 'tsne_noise.jpg'), dpi=300)
    plt.close()

    fig = plt.figure(figsize=(7.1, 10))
    grid = ImageGrid(fig, 111,  # as in plt.subplot(111)
                     nrows_ncols=(4, 2),
                     axes_pad=0.5,
                     share_all=True,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="7%",
                     cbar_pad=0.15)
    for ax, (label, group) in zip(grid, arch_ranking_group_by):
        ax.set_title(label.upper())
        ax.set_ylim((y_min, y_max))

        ax.set_xlim((x_min, x_max))
        sc = ax.scatter(group.X_emb_0.to_numpy(), group.X_emb_1.to_numpy(), s=2, alpha=0.15,
                        c=group.arch_ranking.to_numpy(), cmap=plt.get_cmap('viridis'))
        ax.grid(True, which="both", ls="-", alpha=0.1)
    # Colorbar
    ax.cax.colorbar(sc)
    ax.cax.toggle_label(True)
    ax.cax.set_ylabel('Architecture noise ranking', rotation=270, labelpad=15)
    #plt.tight_layout()
    plt.savefig(os.path.join(model_log_dir, 'tsne_noise_groupby_algorithms.jpg'), dpi=600)
    plt.close()

    # Comparison between the different optimizers
    for label, group in arch_ranking_group_by:
        plt.figure(figsize=(4, 3))
        plt.xlabel('t-SNE 1st component')
        plt.ylabel('t-SNE 2nd component')
        plt.ylim((y_min, y_max))
        plt.xlim((x_min, x_max))
        plt.scatter(group.X_emb_0, group.X_emb_1, s=2, alpha=0.15, c=group.arch_ranking, cmap=plt.get_cmap('viridis'))
        plt.clim(np.min(arch_ranking), np.max(arch_ranking))
        if label == 'rs':
            cbar = plt.colorbar()
            cbar.ax.set_ylabel('Noise ranking', rotation=270, labelpad=15)
        plt.grid(True, which="both", ls="-", alpha=0.5)
        #plt.tight_layout()
        plt.savefig(os.path.join(model_log_dir, 'tsne_groupby_algorithms_{}.jpg'.format(label)), dpi=300)
        plt.close()
    rcParams.update({'figure.autolayout': True})


if __name__ == "__main__":
    plotting()
