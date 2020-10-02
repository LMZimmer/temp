from datetime import datetime
from pathlib import Path

import click
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.manifold import TSNE
from tqdm import tqdm

from analysis.global_nasbench_plots import compute_depth_of_cell_simple_path, compute_no_weight_operations
from surrogate_models.utils import ConfigLoader

sns.set_style('whitegrid')

rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'font.size': 14})

fontP = FontProperties()
fontP.set_size('small')


@click.command()
@click.option('--nasbench_data', type=click.STRING, help='path to nasbench root directory')
def plotting(nasbench_data):
    current_time = datetime.now()
    config_loader = ConfigLoader('configspace.json')
    results_paths = [filename for filename in Path(nasbench_data).rglob('*.json')]  # + \
    config_vecs = []
    val_errors = []
    optimizer_name = []
    normal_cell_depths = []
    reduce_cell_depths = []
    num_parameters = []
    num_skip_connect_normal = []
    for config_path in tqdm(results_paths, desc='Reading dataset'):
        if any([name in str(config_path) for name in ["surrogate_model", "de_multi_seed", "only_skip_connect",
                                                      "only_avg_pool_3x3", "only_max_pool_3x3",
                                                      "cell_topology_analysis"]]):
            continue
        try:
            config_space_instance, val_accuracy, test_accuracy, json_file = config_loader[config_path]
            optimizer = config_path.parent.stem
            if optimizer in ['darts', 'bananas', 'combo', 'de', 're', 'tpe', 'random_ws', 'pc_darts', 'gdas', 'rs',
                             'drnas']:
                optimizer_name.append(optimizer)
                config_vec = config_space_instance.get_array()
                config_vecs.append(config_vec)
                normal_depth, reduce_depth = compute_depth_of_cell_simple_path(config_space_instance)
                normal_cell_depths.append(normal_depth)
                reduce_cell_depths.append(reduce_depth)
                num_skip_connect_normal.append(
                    compute_no_weight_operations(config_space_instance)['normal']['skip_connect'])
                num_parameter = json_file['info'][0]['model_parameters']
                num_parameters.append(num_parameter)
                val_accuracy = json_file['info'][0]['val_accuracy']
                val_errors.append(1 - val_accuracy / 100)
        except Exception as e:
            print('error', e, config_path)

    arch_ranking = ss.rankdata(val_errors)
    val_errors_group_by = pd.DataFrame(
        {'optimizer': optimizer_name,
         'val_error': val_errors,
         'cell_depth': normal_cell_depths}).groupby('optimizer')

    # ALGORITHM ECDF PLOTS
    fig = plt.figure(figsize=(4, 3))
    for name, group in val_errors_group_by:
        plt.plot(np.sort(group.val_error.to_numpy()),
                 np.linspace(0, 1, len(group.val_error.to_numpy()), endpoint=False),
                 label=name.upper(), linewidth=2)
    plt.gca().tick_params(axis='both', which='minor', labelsize=8)
    plt.legend(prop={'size': 7})
    plt.xscale('log')
    plt.xlabel('Validation error')
    plt.ylabel('Cumulative prob.')
    plt.xticks([0.05, 0.06, 0.1, 0.2], ["0.05", "", "0.1", "0.2"])
    plt.yticks(np.linspace(0, 1.0, 5))
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig('analysis/plot_export/algorithm_val_error_ecdf.pdf', dpi=200)
    plt.close()

    # OPTIMIZER VAL ERROR (VIOLINPLOT)
    cell_depth_optimizer = pd.DataFrame(
        {'optimizer': optimizer_name,
         'val_error': val_errors})
    plt.figure(figsize=(4, 3))
    sns.violinplot(x=[name.upper() for name in cell_depth_optimizer['optimizer']],
                   y=cell_depth_optimizer['val_error'], palette="Set2", linewidth=0.1, cut=0)
    plt.xticks(rotation=70)
    plt.ylabel('Validation Error')
    plt.savefig('analysis/plot_export/algorithm_comp_violin.pdf', dpi=200)
    plt.close()

    # NORMAL / REDUCTION CELL DEPTH VS OPTIMIZER (VIOLINPLOT)
    cell_depth_optimizer = pd.DataFrame(
        {'optimizer': [*optimizer_name, *optimizer_name],
         'val_error': [*val_errors, *val_errors],
         'Cell Depth': [*normal_cell_depths, *reduce_cell_depths],
         'Cell Type': [*['normal' for _ in normal_cell_depths],
                       *['reduce' for _ in reduce_cell_depths]]})
    plt.figure(figsize=(4, 3))
    sns.violinplot(x=[name.upper() for name in cell_depth_optimizer['optimizer']],
                   y=cell_depth_optimizer['Cell Depth'], hue=cell_depth_optimizer['Cell Type'], split=True,
                   palette="Set2", linewidth=0.5, cut=0)
    plt.xticks(rotation=70)
    plt.ylabel('Cell Depth')
    plt.savefig('analysis/plot_export/algorithm_comp_depth_violin.pdf', dpi=200)
    plt.close()

    print('########## STARTING TSNE')
    X_no_nan = np.nan_to_num(np.array(config_vecs), nan=0.0)
    X_emb = TSNE(n_components=2).fit_transform(X_no_nan)
    X_emb_group_by = np.split(X_emb, np.cumsum(np.unique(optimizer_name, return_counts=True)[1])[:-1])
    arch_ranking_group_by = pd.DataFrame(
        {'optimizer': optimizer_name, 'arch_ranking': arch_ranking, 'X_emb_0': X_emb[:, 0],
         'X_emb_1': X_emb[:, 1]}).groupby('optimizer')

    y_min, y_max, x_min, x_max = np.min(X_emb[:, 1]), np.max(X_emb[:, 1]), np.min(X_emb[:, 0]), np.max(X_emb[:, 0])

    # TSNE VAL ERRORS
    plt.figure(figsize=(4, 3))
    # plt.title('NAS-Bench-201 {}, num. models: {}'.format(current_time.strftime("%b %d %Y %H:%M"), len(test_errors)))
    plt.xlabel(' 1st component')
    plt.ylabel(' 2nd component')
    ax = plt.gca()
    plt.scatter(X_emb[:, 0], X_emb[:, 1], s=2, alpha=0.15, c=val_errors, norm=matplotlib.colors.LogNorm(),
                cmap=plt.get_cmap('viridis'))
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Validation Error', rotation=270, labelpad=15)
    cbar.set_ticks([0.05, 0.06, 0.1, 0.2])
    cbar.set_ticklabels(["0.05", "", "0.1", "0.2"])
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig('analysis/plot_export/tsne_val_error.png', dpi=300)
    plt.close()

    # TSNE SKIP CONNECTIONS
    plt.figure(figsize=(4, 3))
    # plt.title('NAS-Bench-201 {}, num. models: {}'.format(current_time.strftime("%b %d %Y %H:%M"), len(test_errors)))
    plt.xlabel(' 1st component')
    plt.ylabel(' 2nd component')
    ax = plt.gca()
    plt.scatter(X_emb[:, 0], X_emb[:, 1], s=2, alpha=0.15, c=num_skip_connect_normal, cmap=plt.get_cmap('viridis'))
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Num. Skip-Connection Normal', rotation=270, labelpad=15)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig('analysis/plot_export/tsne_skip_connect_normal.png', dpi=300)
    plt.close()

    # TSNE NUM PARAMETERS
    plt.figure(figsize=(4, 3))
    # plt.title('NAS-Bench-201 {}, num. models: {}'.format(current_time.strftime("%b %d %Y %H:%M"), len(test_errors)))
    plt.xlabel(' 1st component')
    plt.ylabel(' 2nd component')
    ax = plt.gca()
    plt.scatter(X_emb[:, 0], X_emb[:, 1], s=2, alpha=0.15, c=num_parameters, norm=matplotlib.colors.LogNorm(),
                cmap=plt.get_cmap('viridis'))
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Num. Parameters', rotation=270, labelpad=15)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig('analysis/plot_export/tsne_num_parameter.png', dpi=300)
    plt.close()

    # TSNE NORMAL CELL DEPTH
    plt.figure(figsize=(4, 3))
    # plt.title('NAS-Bench-201 {}, num. models: {}'.format(current_time.strftime("%b %d %Y %H:%M"), len(test_errors)))
    plt.xlabel(' 1st component')
    plt.ylabel(' 2nd component')
    ax = plt.gca()
    plt.scatter(X_emb[:, 0], X_emb[:, 1], s=2, alpha=0.15, c=normal_cell_depths, cmap=plt.get_cmap('viridis'))
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Normal Cell Depth', rotation=270, labelpad=15)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig('analysis/plot_export/tsne_cell_depth.png', dpi=300)
    plt.close()

    # TSNE ARCHITECTURE RANKING
    plt.figure(figsize=(4.5, 3.1))
    # plt.title('NAS-Bench-201 {}, num. models: {}'.format(current_time.strftime("%b %d %Y %H:%M"), len(test_errors)))
    plt.xlabel('1st component')
    plt.ylabel('2nd component')
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.scatter(X_emb[:, 0], X_emb[:, 1], s=2, alpha=0.15, c=ss.rankdata(val_errors), cmap=plt.get_cmap('viridis'))
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Architecture Ranking', rotation=270, labelpad=15)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig('analysis/plot_export/tsne_arch_ranking.png', dpi=300)
    plt.close()

    rcParams.update({'figure.autolayout': False})

    # TSNE ARCHITECTURE RANKING BY OPTIIZER
    fig = plt.figure(figsize=(12, 12))
    grid = ImageGrid(fig, 111,  # as in plt.subplot(111)
                     nrows_ncols=(4, 3),
                     axes_pad=0.5,
                     share_all=True,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="4%",
                     cbar_pad=0.15)
    for ax, (label, group) in zip(grid, arch_ranking_group_by):
        ax.set_aspect('equal')
        ax.set_title(label.upper())
        ax.set_ylim((y_min, y_max))
        ax.set_xlim((x_min, x_max))
        sc = ax.scatter(group.X_emb_0.to_numpy(), group.X_emb_1.to_numpy(), s=2, alpha=0.15,
                        c=group.arch_ranking.to_numpy(), cmap=plt.get_cmap('viridis'))
        ax.grid(True, which="both", ls="-", alpha=0.1)
    # Colorbar
    ax.cax.colorbar(sc)
    ax.cax.toggle_label(True)
    ax.cax.set_ylabel('Architecture Ranking', rotation=270, labelpad=15)
    plt.tight_layout()
    plt.savefig('analysis/plot_export/tsne_groupby_algorithms.png', dpi=600)
    plt.close()

    # Comparison between the different optimizers
    for label, group in arch_ranking_group_by:
        plt.figure(figsize=(4, 3))
        plt.xlabel(' 1st component')
        plt.ylabel(' 2nd component')
        plt.ylim((y_min, y_max))
        plt.xlim((x_min, x_max))
        plt.scatter(group.X_emb_0, group.X_emb_1, s=2, alpha=0.15, c=group.arch_ranking, cmap=plt.get_cmap('viridis'))
        plt.clim(np.min(arch_ranking), np.max(arch_ranking))
        if label == 'rs':
            cbar = plt.colorbar()
            cbar.ax.set_ylabel('Ranking', rotation=270, labelpad=15)
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.tight_layout()
        plt.savefig('analysis/plot_export/tsne_groupby_algorithms_{}.png'.format(label), dpi=300)
        plt.close()
    rcParams.update({'figure.autolayout': True})

    plt.figure(figsize=(4, 3))
    plt.xlabel(' 1st component')
    plt.ylabel(' 2nd component')
    plt.ylim((y_min, y_max))
    plt.xlim((x_min, x_max))
    for label, group in arch_ranking_group_by:
        plt.scatter(group.X_emb_0, group.X_emb_1, s=2, alpha=0.15, label=label)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('analysis/plot_export/tsne_groupby_algorithms_clustered.png', dpi=300)
    plt.close()


if __name__ == "__main__":
    plotting()
