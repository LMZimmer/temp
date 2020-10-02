from datetime import datetime
from pathlib import Path

import click
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import PathPatch
from scipy import stats
from scipy.stats import kendalltau, spearmanr
from tqdm import tqdm

from surrogate_models.gnn.gnn_utils import NASBenchDataset
from surrogate_models.utils import ConfigLoader

sns.set_style('whitegrid')
# rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'font.size': 14})

nasbench_dataset = NASBenchDataset(None, None, None, None, None)


def compute_depth_of_cell(config_space_instance):
    """
    Compute the depth of a cell following the definition of XNAS
    :param config_space_instance:
    :return:
    """
    config_dict = config_space_instance.get_dictionary()
    cell_depth = {'normal': 0, 'reduce': 0}
    for cell_type in ['normal', 'reduce']:
        parents = []
        for node in range(3, 6):
            parent_0, parent_1 = config_dict[
                'NetworkSelectorDatasetInfo:darts:inputs_node_{}_{}'.format(cell_type, node)].split('_')
            parents.extend([int(parent_0), int(parent_1)])
        cell_depth[cell_type] = np.mean(parents)
    return cell_depth['normal'], cell_depth['reduce']


def compute_depth_of_cell_simple_path(config_space_instance):
    """
    Compute the depth of a cell following the definition of
    :param config_space_instance:
    :return:
    """
    adj_cells = nasbench_dataset.create_darts_adjacency_matrix_from_config(config_space_instance)
    cell_depth = {'normal': 0, 'reduce': 0}
    for cell_type, (adj, ops) in zip(['normal', 'reduce'], adj_cells):
        for i in range(2, 6):
            adj[i, 6] = 1
        dag = nx.DiGraph(adj)
        depth = max(max(
            *[[len(path) for path in list(nx.all_simple_paths(dag, source=source, target=6))] for source in [0, 1]]))
        cell_depth[cell_type] = depth

    return cell_depth['normal'], cell_depth['reduce']


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


def compute_mean_trajectory(performance_list):
    mean_trajectory = []
    for ind in range(len(performance_list)):
        mean_perf_at_ind = np.mean(performance_list[0:ind + 1])
        mean_trajectory.append(mean_perf_at_ind)
    return mean_trajectory


def adjust_box_widths(g, fac):
    """
    Adjust the widths of a seaborn-generated boxplot.
    """

    # iterating through Axes instances
    for ax in g.axes:

        # iterating through axes artists:
        for c in ax.get_children():

            # searching for PathPatches
            if isinstance(c, PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5 * (xmin + xmax)
                xhalf = 0.5 * (xmax - xmin)

                # setting new width of box
                xmin_new = xmid - fac * xhalf
                xmax_new = xmid + fac * xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])


@click.command()
@click.option('--nasbench_data', type=click.STRING, help='path to nasbench root directory')
def plotting(nasbench_data):
    current_time = datetime.now()
    config_loader = ConfigLoader('configspace.json')
    model_parameters = []

    config_space_instances = []
    normal_depths = []
    reduce_depths = []
    train_errors = []
    test_errors = []
    val_errors = []
    num_layers_model = []
    init_channels_model = []
    num_epochs = []
    learning_rates = []
    run_times = []
    batch_sizes = []
    cutout_lengths = []
    weight_decays = []
    normal_cell_no_ops = []
    no_weight_ops_list = []
    results_paths = [filename for filename in Path(nasbench_data).rglob('*.json')]  # + \
    # glob.glob(os.path.join(nasbench_data, 'random_fidelities/results_fidelity_*/results_*.json'))

    for config_path in tqdm(results_paths, desc='Reading dataset'):
        # for config_path in tqdm(results_paths[0:5000], desc='Reading dataset'):
        try:
            config_space_instance, val_accuracy, test_accuracy, json_file = config_loader[config_path]
            config_space_instances.append(config_space_instance)
            normal_depth, reduce_depth = compute_depth_of_cell_simple_path(config_space_instance)
            no_weight_ops = compute_no_weight_operations(config_space_instance)
            no_weight_ops_list.append(no_weight_ops)
            normal_depths.append(normal_depth)
            reduce_depths.append(reduce_depth)
            test_accuracy = json_file['test_accuracy']
            config_space_instances.append(config_space_instance.get_dictionary())

            num_parameters = json_file['info'][0]['model_parameters']
            epochs = json_file['budget']
            train_accuracy = json_file['info'][0]['train_accuracy_final']
            val_accuracy = json_file['info'][0]['val_accuracy']

            num_layers = json_file['optimized_hyperparamater_config']['NetworkSelectorDatasetInfo:darts:layers']
            init_channels = json_file['optimized_hyperparamater_config'][
                'NetworkSelectorDatasetInfo:darts:init_channels']
            run_time = json_file['runtime']
            learning_rate = json_file['optimized_hyperparamater_config']['OptimizerSelector:sgd:learning_rate']
            batch_size = json_file['optimized_hyperparamater_config']['CreateImageDataLoader:batch_size']
            cutout_length = json_file['optimized_hyperparamater_config']['ImageAugmentation:cutout_length']
            weight_decay = json_file['optimized_hyperparamater_config']['OptimizerSelector:sgd:weight_decay']

            weight_decays.append(weight_decay)
            cutout_lengths.append(cutout_length)
            run_times.append(run_time)
            model_parameters.append(num_parameters)
            num_epochs.append(epochs)
            batch_sizes.append(batch_size)

            train_errors.append(1 - train_accuracy / 100)
            test_errors.append(1 - test_accuracy / 100)
            val_errors.append(1 - val_accuracy / 100)

            num_layers_model.append(num_layers)
            init_channels_model.append(init_channels)
            learning_rates.append(learning_rate)

        except Exception as e:
            print('error', e, config_path, json_file)

    # Compute general statistic
    result_statistics = {
        'validation_error': [np.mean(val_errors), np.std(val_errors)],
        'test_error': [np.mean(test_errors), np.std(test_errors)],
        'learning_rate': [np.mean(learning_rates), np.std(learning_rates)],
        'num_layers': [np.mean(num_layers_model), np.std(num_layers_model)],
        'epochs': [np.mean(num_epochs), np.std(num_epochs)],
        'init_channels': [np.mean(init_channels_model), np.std(init_channels_model)],
        'weight_decay': [np.mean(weight_decays), np.std(weight_decays)],
    }

    # NUM UNIQUE CONFIGS:
    config_hashdict = {hash(config.__repr__()): config for config in config_space_instances}
    config_hashes = [hash(config.__repr__()) for config in config_space_instances]
    config_occurences = {config_hash: config_hashes.count(config_hash) for config_hash in set(config_hashes)}
    max_occurences = max(config_occurences.values())

    occurence_counts = list(config_occurences.values())
    occurence_counts.sort(reverse=True)

    num_max_occurences = 0
    most_frequent_hashses = []
    most_frequent_configs = []

    config_hash_to_performances = dict()

    for config, performance, parameters in zip(config_space_instances, val_errors, model_parameters):
        config_hash = hash(config.__repr__())
        if config_hash not in config_hash_to_performances.keys():
            config_hash_to_performances[config_hash] = []
        val_acc = (1 - performance) * 100
        config_hash_to_performances[config_hash].append((val_acc, parameters))

    vals_and_pars = list(config_hash_to_performances.values())
    vals_and_pars.sort(key=lambda x: len(x), reverse=True)
    duplicate_performances = []
    duplicate_parameters = []
    for vals_and_pars_list in vals_and_pars:
        perf_list, par_list = map(list, zip(*vals_and_pars_list))
        duplicate_performances.append(perf_list)
        duplicate_parameters.append(par_list)

    for chash, occurences in config_occurences.items():
        if occurences == max_occurences:
            num_max_occurences += 1
            most_frequent_configs.append(config_hashdict[chash])
            most_frequent_hashses.append(chash)

    val_acc_means = [np.mean(performance_list) for performance_list in duplicate_performances if
                     len(performance_list) > 4]
    val_acc_stddevs = [np.std(performance_list) for performance_list in duplicate_performances if
                       len(performance_list) > 4]
    model_parameters_reduced = [parameter_list[0] for parameter_list in duplicate_parameters if len(parameter_list) > 4]

    #most_occuring_config_inds = np.where(np.array(config_hashes) == most_frequent_hashses[0])[0]
    #most_occuring_config_errors = np.array(val_errors)[most_occuring_config_inds]

    # ACCURACY HISTOGRAM (MOST OCCURING CONFIG)
    #plt.figure(figsize=(4, 3))
    #plt.hist(most_occuring_config_errors, bins=10, label='Validation error', alpha=0.4)
    #plt.xlabel('Validation error')
    ## plt.yscale('log')
    #plt.grid(True, which="both", ls="-", alpha=0.5)
    #plt.legend()
    #plt.savefig('analysis/plot_export/val_error_histogram.pdf', dpi=300)

    # NUMBER EVALUATIONS VS MEAN ACCURACY (TOP N)
    top_n = 20
    min_num_evals = 5
    mean_accuracy_trajectories = [compute_mean_trajectory(performance_list) for performance_list in
                                  duplicate_performances]
    mean_error_at_1_evals = [abs(acc_trajectory[0] - acc_trajectory[-1]) for acc_trajectory in
                             mean_accuracy_trajectories if len(acc_trajectory) >= min_num_evals]
    mean_error_at_3_evals = [abs(acc_trajectory[2] - acc_trajectory[-1]) for acc_trajectory in
                             mean_accuracy_trajectories if len(acc_trajectory) >= min_num_evals]
    # mean_error_at_10_evals = [abs(acc_trajectory[10]-acc_trajectory[-1]) for acc_trajectory in mean_accuracy_trajectories if len(acc_trajectory)>10]
    mean_error_at_1_evals_rel = [abs(acc_trajectory[0] - acc_trajectory[-1]) / acc_trajectory[-1] for acc_trajectory in
                                 mean_accuracy_trajectories if len(acc_trajectory) >= min_num_evals]
    mean_error_at_3_evals_rel = [abs(acc_trajectory[2] - acc_trajectory[-1]) / acc_trajectory[-1] for acc_trajectory in
                                 mean_accuracy_trajectories if len(acc_trajectory) >= min_num_evals]
    # mean_error_at_10_evals_rel = [abs(acc_trajectory[10]-acc_trajectory[-1])/acc_trajectory[-1] for acc_trajectory in mean_accuracy_trajectories if len(acc_trajectory)>10]
    print("==> Mean error made when estimating the mean accuracy with only 1 seed is {:.2f}".format(
        np.mean(mean_error_at_1_evals)))
    print("==> Mean error made when estimating the mean accuracy with only 3 seeds is {:.2f}".format(
        np.mean(mean_error_at_3_evals)))
    # print("==> Mean error made when estimating the mean accuracy with only 10 seeds is {:.2f}".format(np.mean(mean_error_at_10_evals)))
    print("==> Mean error made when estimating the mean accuracy with only 1 seed is {:.2%}".format(
        np.mean(mean_error_at_1_evals_rel)))
    print("==> Mean error made when estimating the mean accuracy with only 3 seeds is {:.2%}".format(
        np.mean(mean_error_at_3_evals_rel)))
    # print("==> Mean error made when estimating the mean accuracy with only 10 seeds is {:.2%}".format(np.mean(mean_error_at_10_evals_rel)))
    plt.figure(figsize=(4, 3))
    for ind in range(top_n):
        yy = mean_accuracy_trajectories[ind][0:20]
        xx = np.arange(len(yy))
        plt.plot(xx, yy, ls='--')
    plt.xlabel('Num evaluations')
    plt.ylabel('Mean val. error')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.savefig('analysis/plot_export/num_evals_vs_mean_error.pdf', dpi=300)

    # PARAMETERS VS. VAL ACC / VAL ACC STD
    model_parameters_scaled = np.array(model_parameters_reduced) / 10 ** 6
    plt.figure(figsize=(4, 3))
    plt.scatter(model_parameters_scaled, val_acc_stddevs, s=2, alpha=0.3, c=val_acc_means, cmap=plt.get_cmap('magma_r'))
    plt.ylabel('Validation accuracy')
    plt.xlabel('Num. parameters [10e6]')
    ax = plt.gca()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.xlim(0.7, 2.05)
    plt.ylim(89.5, 95.5)
    # plt.xticks([0.7, 1.0, 2.0], ["", "1", "2"])
    plt.yticks([90, 95], ["90", "95"])
    # plt.yticks([0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11], ["0.05", "", "", "", "", "0.1", ""])
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Standard deviavtion', rotation=270, labelpad=15)
    plt.tight_layout()
    plt.savefig('analysis/plot_export/parameters_vs_val_acc_vs_stddev.png', dpi=300)
    plt.close()

    # VAL ACC STD VS. VAL ACC / PARAMETERS
    model_parameters_scaled = np.array(model_parameters_reduced) / 10 ** 6
    plt.figure(figsize=(4, 3))
    val_acc_stddevs_01 = [vac/100 for vac in val_acc_stddevs]
    val_acc_means_01 = [vstd/100 for vstd in val_acc_means]
    plt.scatter(val_acc_stddevs_01, val_acc_means_01, s=2, alpha=0.3, c=model_parameters_scaled,
                #norm=matplotlib.colors.LogNorm(),
                cmap=plt.get_cmap('magma_r'))
    plt.ylabel('Validation accuracy')
    plt.xlabel('Standard deviation')
    ax = plt.gca()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    #plt.xlim(0.7, 2.05)
    plt.ylim(89.5/100, 95.5/100)
    #plt.xticks([0.7, 1.0, 2.0], ["", "1", "2"])
    plt.yticks([0.9, 0.95], ["0.9", "0.95"])
    #plt.yticks([0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11], ["0.05", "", "", "", "", "0.1", ""])
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Num. parameters [10e6]', rotation=270, labelpad=15)
    plt.clim(0.7, 2.05)
    cbar.set_ticks([0.7, 1.0, 2.0])
    cbar.set_ticklabels(["", "1", "2"])
    plt.tight_layout()
    plt.savefig('analysis/plot_export/stddev_vs_val_acc_vs_parameters.png', dpi=300)
    plt.close()

    # VALIDATION ERROR VS CELL DEPTH
    bin_means_normal, bin_edges_normal, binnumber_normal = stats.binned_statistic(val_errors, normal_depths, 'mean',
                                                                                  bins=30)
    bin_stds_normal, bin_edges_normal, _ = stats.binned_statistic(val_errors, normal_depths, 'std', bins=30)
    bin_means_reduce, bin_edges_reduce, binnumber_reduce = stats.binned_statistic(val_errors, reduce_depths, 'mean',
                                                                                  bins=30)
    bin_stds_reduce, bin_edges_reduce, _ = stats.binned_statistic(val_errors, reduce_depths, 'std', bins=30)
    plt.figure(figsize=(4, 3))
    plt.hlines(bin_means_normal, bin_edges_normal[:-1], bin_edges_normal[1:], colors='red', label='Normal cell depth')
    plt.hlines(bin_means_reduce, bin_edges_reduce[:-1], bin_edges_reduce[1:], colors='blue',
               label='Reduction cell depth')
    plt.xlabel('Validation error')
    plt.ylabel('Cell depth')
    for idx, (x_min, x_max, bin_n) in enumerate(zip(bin_edges_normal[:-1], bin_edges_normal[1:], binnumber_normal)):
        xs = np.linspace(x_min, x_max, 30)

        std = bin_stds_normal[idx]
        if std != 0:
            std = bin_stds_normal[idx] / np.sqrt(bin_n)
        plt.fill_between(x=xs, y1=bin_means_normal[idx] - std, y2=bin_means_normal[idx] + std, alpha=0.1, color='red')

    for idx, (x_min, x_max, bin_n) in enumerate(zip(bin_edges_reduce[:-1], bin_edges_reduce[1:], binnumber_reduce)):
        xs = np.linspace(x_min, x_max, 30)
        std = bin_stds_normal[idx]
        if std != 0:
            std = bin_stds_reduce[idx] / np.sqrt(bin_n)
        plt.fill_between(x=xs, y1=bin_means_reduce[idx] - std, y2=bin_means_reduce[idx] + std, alpha=0.1, color='blue')

    plt.xscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend(loc='upper left')
    plt.savefig('analysis/plot_export/val_error_vs_normal_and_reduce_cell_depth.pdf', dpi=300)
    plt.close()

    # VALIDATION ERROR VS OPERATIONS
    for op in ['max_pool_3x3', 'avg_pool_3x3', 'skip_connect']:
        num_no_op_normal = [cell['normal'][op] for cell in no_weight_ops_list]
        num_no_op_reduce = [cell['reduce'][op] for cell in no_weight_ops_list]
        bin_means_normal, bin_edges_normal, binnumber_normal = \
            stats.binned_statistic(val_errors, num_no_op_normal, 'mean', bins=30)
        bin_stds_normal, bin_edges_normal, _ = \
            stats.binned_statistic(val_errors, num_no_op_normal, 'std', bins=30)
        bin_means_reduce, bin_edges_reduce, binnumber_reduce = \
            stats.binned_statistic(val_errors, num_no_op_reduce, 'mean', bins=30)
        bin_stds_reduce, bin_edges_reduce, _ = \
            stats.binned_statistic(val_errors, num_no_op_reduce, 'std', bins=30)
        plt.figure(figsize=(4, 3))
        plt.hlines(bin_means_normal, bin_edges_normal[:-1], bin_edges_normal[1:], colors='red', label='Normal Cell')
        plt.hlines(bin_means_reduce, bin_edges_reduce[:-1], bin_edges_reduce[1:], colors='blue',
                   label='Reduction cell')
        plt.xlabel('Validation error')
        plt.ylabel('Num. {}'.format(op).replace("_", " "))

        for idx, (x_min, x_max, bin_n) in enumerate(zip(bin_edges_normal[:-1], bin_edges_normal[1:], binnumber_normal)):
            xs = np.linspace(x_min, x_max, 30)

            std = bin_stds_normal[idx]
            if std != 0:
                std = bin_stds_normal[idx] / np.sqrt(bin_n)
            plt.fill_between(x=xs, y1=bin_means_normal[idx] - std, y2=bin_means_normal[idx] + std, alpha=0.1,
                             color='red')

        for idx, (x_min, x_max, bin_n) in enumerate(zip(bin_edges_reduce[:-1], bin_edges_reduce[1:], binnumber_reduce)):
            xs = np.linspace(x_min, x_max, 30)
            std = bin_stds_normal[idx]
            if std != 0:
                std = bin_stds_reduce[idx] / np.sqrt(bin_n)
            plt.fill_between(x=xs, y1=bin_means_reduce[idx] - std, y2=bin_means_reduce[idx] + std, alpha=0.1,
                             color='blue')

        plt.xscale('log')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.legend()
        plt.savefig('analysis/plot_export/val_error_vs_normal_and_reduce_num_{}.pdf'.format(op), dpi=300)
        plt.close()

    # CELL DEPTH HISTOGRAM
    plt.figure(figsize=(4, 3))
    plt.hist(normal_depths, bins=30, label='Normal Cell Depth', alpha=0.4)
    plt.hist(reduce_depths, bins=30, label='Reduction Cell Depth', alpha=0.4)
    plt.xlabel('Cell Depth')
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.savefig('analysis/plot_export/normal_reduce_histogram.pdf', dpi=300)

    print('corr val test (kendall):', kendalltau(val_errors, test_errors))
    print('corr train val (kendall):', kendalltau(train_errors, val_errors))

    print('corr val test (spearman):', spearmanr(val_errors, test_errors))
    print('corr train val (spearman):', spearmanr(train_errors, val_errors))

    print('total runtime:', np.sum(run_times))
    print('best performance (test accuracy):', 100 - np.min(test_errors) * 100)
    print('dataset statistics', result_statistics)

    # CELL DEPTH VS VALIDATION ERROR
    data_dict = {"Cell type": ["normal"] * len(normal_depths) + ["reduction"] * len(reduce_depths),
                 "Cell depth": normal_depths + reduce_depths,
                 "Num max pool": [cell["normal"]["max_pool_3x3"] for cell in no_weight_ops_list] + [
                     cell["reduce"]["max_pool_3x3"] for cell in no_weight_ops_list],
                 "Num skip conn.": [cell["normal"]["skip_connect"] for cell in no_weight_ops_list] + [
                     cell["reduce"]["skip_connect"] for cell in no_weight_ops_list],
                 "Num avg pool": [cell["normal"]["avg_pool_3x3"] for cell in no_weight_ops_list] + [
                     cell["reduce"]["avg_pool_3x3"] for cell in no_weight_ops_list],
                 "Validation error": val_errors * 2}
    df = pd.DataFrame(data=data_dict)

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.violinplot(x="Cell depth", y="Validation error", hue="Cell type", data=df, palette="Set2", linewidth=0.1,
                   split=True, cut=0)

    ax.set_yscale("log")
    ax.set_yticks([0.05, 0.06, 0.1, 0.2])
    ax.set_yticklabels(["0.05", "", "0.1", "0.2"])
    plt.grid(True, which="both", ls="-", axis="y", alpha=0.5)
    plt.legend(loc="upper right")
    plt.savefig('analysis/plot_export/val_error_vs_normal_and_reduce_cell_depth_violin.pdf', dpi=300)
    plt.close()

    # NORMAL NUM OPERATIONS VS VALIDATION ERROR (SKIP CONN., MAX POOL, AVG POOL)
    num_avg_pool = [cell["normal"]["avg_pool_3x3"] for cell in no_weight_ops_list]
    num_max_pool = [cell["normal"]["max_pool_3x3"] for cell in no_weight_ops_list]
    num_skip_conn = [cell["normal"]["skip_connect"] for cell in no_weight_ops_list]

    data_dict = {"Num operations": num_avg_pool + num_max_pool + num_skip_conn,
                 "Operation type": ["Num avg pool"] * len(num_avg_pool) + ["Num max pool"] * len(num_max_pool) + [
                     "Num skip conn."] * len(num_skip_conn),
                 "Validation error": val_errors * 3}
    df = pd.DataFrame(data=data_dict)

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.violinplot(x="Num operations", y="Validation error", hue="Operation type", data=df, linewidth=0.1, cut=0)
    # sns.boxplot(x="Num operations", y="Validation error", hue="Operation type", data=df, linewidth=1, ax=ax)

    ax.set_yscale("log")
    ax.set_ylim([0.045, 0.25])
    ax.set_yticks([0.05, 0.06, 0.1, 0.2])
    ax.set_yticklabels(["0.05", "", "0.1", "0.2"])
    plt.grid(True, which="both", ls="-", axis="y", alpha=0.5)
    plt.legend(loc="upper left", prop={'size': 12})
    # adjust_box_widths(fig, 0.9)
    plt.savefig('analysis/plot_export/val_error_vs_operation_normal.pdf', dpi=300)
    plt.close()

    # REDUCTION NUM OPERATIONS VS VALIDATION ERROR (SKIP CONN., MAX POOL, AVG POOL)
    num_avg_pool = [cell["reduce"]["avg_pool_3x3"] for cell in no_weight_ops_list]
    num_max_pool = [cell["reduce"]["max_pool_3x3"] for cell in no_weight_ops_list]
    num_skip_conn = [cell["reduce"]["skip_connect"] for cell in no_weight_ops_list]

    data_dict = {"Num operations": num_avg_pool + num_max_pool + num_skip_conn,
                 "Operation type": ["Num avg pool"] * len(num_avg_pool) + ["Num max pool"] * len(num_max_pool) + [
                     "Num skip conn."] * len(num_skip_conn),
                 "Validation error": val_errors * 3}
    df = pd.DataFrame(data=data_dict)

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.violinplot(x="Num operations", y="Validation error", hue="Operation type", data=df, linewidth=0.1, cut=0)
    # sns.boxplot(x="Num operations", y="Validation error", hue="Operation type", data=df, linewidth=1, ax=ax)

    ax.set_yscale("log")
    ax.set_ylim([0.045, 0.25])
    ax.set_yticks([0.05, 0.06, 0.1, 0.2])
    ax.set_yticklabels(["0.05", "", "0.1", "0.2"])
    plt.grid(True, which="both", ls="-", axis="y", alpha=0.5)
    plt.legend(loc="upper left", prop={'size': 12})
    # adjust_box_widths(fig, 0.9)
    plt.savefig('analysis/plot_export/val_error_vs_operation_reduction.pdf', dpi=300)
    plt.close()

    # NORMAL CELL DEPTH VS REDUCTION CELL DEPTH / VAL ERRORs
    plt.figure(figsize=(4, 3))
    # plt.title('NAS-Bench-201 {}, num. models: {}'.format(current_time.strftime("%b %d %Y %H:%M"), len(test_errors)))
    plt.ylabel('Normal Cell Depth')
    plt.xlabel('Reduction Cell Depth')
    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.xlim(min(normal_depths), max(normal_depths))
    plt.scatter(normal_depths, reduce_depths, s=2, alpha=0.15, c=val_errors, norm=matplotlib.colors.LogNorm(),
                cmap=plt.get_cmap('magma_r'))
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Validation Error', rotation=270, labelpad=15)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig('analysis/plot_export/normal_reduce_depth.png', dpi=300)
    plt.close()

    # NORMAL CELL DEPTH VS VAL ERROR / REDUCE CELL DEPTH
    plt.figure(figsize=(4, 3))
    # plt.title('NAS-Bench-201 {}, num. models: {}'.format(current_time.strftime("%b %d %Y %H:%M"), len(test_errors)))
    plt.ylabel('Validation Error')
    plt.xlabel('Normal Cell Depth')
    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.xlim(min(normal_depths), max(normal_depths))
    plt.scatter(normal_depths, val_errors, s=2, alpha=0.15, c=reduce_depths, norm=matplotlib.colors.LogNorm(),
                cmap=plt.get_cmap('magma_r'))
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Reduction Cell Depth', rotation=270, labelpad=15)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig('analysis/plot_export/normal_depth_val_error.png', dpi=300)
    plt.close()

    # REDUCTION CELL DEPTH VS VAL ERROR / NORMAL CELL DEPTH
    plt.figure(figsize=(4, 3))
    # plt.title('NAS-Bench-201 {}, num. models: {}'.format(current_time.strftime("%b %d %Y %H:%M"), len(test_errors)))
    plt.ylabel('Validation Error')
    plt.xlabel('Reduction Cell Depth')
    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.xlim(min(normal_depths), max(normal_depths))
    plt.scatter(reduce_depths, val_errors, s=2, alpha=0.15, c=normal_depths, norm=matplotlib.colors.LogNorm(),
                cmap=plt.get_cmap('magma_r'))
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Normal Cell Depth', rotation=270, labelpad=15)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig('analysis/plot_export/reduction_depth_val_error.png', dpi=300)
    plt.close()

    # PARAMETERS VS. VAL ERROR / RUNTIMES
    plt.figure(figsize=(4, 3))
    # plt.title('NAS-Bench-201 {}, num. models: {}'.format(current_time.strftime("%b %d %Y %H:%M"), len(test_errors)))
    plt.ylabel('Validation error')
    plt.xlabel('Num. parameters [10e6]')
    ax = plt.gca()
    ax.set_yscale('log')
    # ax.set_xscale('log')
    model_parameters_scaled = np.array(model_parameters) / 10 ** 6
    plt.xlim(min(model_parameters_scaled)-0.05, max(model_parameters_scaled)+0.05)
    plt.yticks([0.05, 0.06, 0.1, 0.2], ["0.05", "", "0.1", "0.2"])
    plt.scatter(model_parameters_scaled, val_errors, s=2, alpha=0.15, c=run_times, norm=matplotlib.colors.LogNorm(),
                cmap=plt.get_cmap('magma_r'))
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Runtime (s)', rotation=270, labelpad=15)
    cbar.set_ticks([10 ** 4, 2 * 10 ** 3])
    # cbar.set_ticklabels([])
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig('analysis/plot_export/parameters_vs_val_error.png', dpi=400)
    plt.close()

    # NORMAL CELL DEPTH VS NUM PARAMS / VAL ERROR
    plt.figure(figsize=(4, 3))
    # plt.title('NAS-Bench-201 {}, num. models: {}'.format(current_time.strftime("%b %d %Y %H:%M"), len(test_errors)))
    plt.xlabel('Normal Cell Depth')
    plt.ylabel('Num. Parameters')
    ax = plt.gca()
    ax.set_yscale('log')
    plt.scatter(normal_depths, model_parameters, s=2, alpha=0.15, c=val_errors, norm=matplotlib.colors.LogNorm(),
                cmap=plt.get_cmap('magma_r'))
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Validation Error', rotation=270, labelpad=15)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig('analysis/plot_export/normal_depth_num_parameters.png', dpi=300)
    plt.close()

    # REDUCTION CELL DEPTH VS NUM PARAMS / VAL ERROR
    plt.figure(figsize=(4, 3))
    # plt.title('NAS-Bench-201 {}, num. models: {}'.format(current_time.strftime("%b %d %Y %H:%M"), len(test_errors)))
    plt.xlabel('Reduction Cell Depth')
    plt.ylabel('Num. Parameters')
    ax = plt.gca()
    ax.set_yscale('log')
    plt.xlim(0.4, 2.6)
    plt.scatter(reduce_depths, model_parameters, s=2, alpha=0.15, c=val_errors, norm=matplotlib.colors.LogNorm(),
                cmap=plt.get_cmap('magma_r'))
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Validation Error', rotation=270, labelpad=15)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig('analysis/plot_export/reduction_depth_num_parameters.png', dpi=300)
    plt.close()

    # PARAMETERS VS. VAL ERROR / NORMAL CELL DEPTH
    plt.figure(figsize=(4, 3))
    # plt.title('NAS-Bench-201 {}, num. models: {}'.format(current_time.strftime("%b %d %Y %H:%M"), len(test_errors)))
    plt.ylabel('Validation error')
    plt.xlabel('Num. Parameters')
    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.xlim(min(model_parameters), max(model_parameters))
    plt.scatter(model_parameters, val_errors, s=2, alpha=0.15, c=normal_depths, norm=matplotlib.colors.LogNorm(),
                cmap=plt.get_cmap('magma_r'))
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Normal Cell Depth', rotation=270, labelpad=15)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig('analysis/plot_export/parameters_vs_val_error_normal_cell_depth.png', dpi=300)
    plt.close()

    # PARAMETERS VS. VAL ERROR / REDUCTION CELL DEPTH
    plt.figure(figsize=(4, 3))
    # plt.title('NAS-Bench-201 {}, num. models: {}'.format(current_time.strftime("%b %d %Y %H:%M"), len(test_errors)))
    plt.ylabel('Validation error')
    plt.xlabel('Num. Parameters')
    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.xlim(min(model_parameters), max(model_parameters))
    plt.scatter(model_parameters, val_errors, s=2, alpha=0.15, c=reduce_depths, norm=matplotlib.colors.LogNorm(),
                cmap=plt.get_cmap('magma_r'))
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Reduction Cell Depth', rotation=270, labelpad=15)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig('analysis/plot_export/parameters_vs_val_error_reduce_cell_depth.png', dpi=300)
    plt.close()

    # RUNTIME VS. VAL ERROR
    plt.figure(figsize=(4, 3))
    # plt.title('NAS-Bench-201 {}, num. models: {}'.format(current_time.strftime("%b %d %Y %H:%M"), len(test_errors)))
    plt.ylabel('Validation error')
    plt.xlabel('Runtime (s)')
    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.xlim(min(run_times), max(run_times))
    plt.scatter(run_times, val_errors, s=2, alpha=0.15)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig('analysis/plot_export/runtimes_vs_val_error.png', dpi=300)
    plt.close()

    # RUNTIME VS. VAL ERROR / NORMAL CELL DEPTH
    plt.figure(figsize=(4, 3))
    # plt.title('NAS-Bench-201 {}, num. models: {}'.format(current_time.strftime("%b %d %Y %H:%M"), len(test_errors)))
    plt.ylabel('Validation error')
    plt.xlabel('Runtime (s)')
    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.xlim(min(run_times), max(run_times))
    plt.scatter(run_times, val_errors, s=2, alpha=0.15, c=normal_depths, norm=matplotlib.colors.LogNorm(),
                cmap=plt.get_cmap('magma_r'))
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Normal Cell Depth', rotation=270, labelpad=15)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig('analysis/plot_export/runtimes_vs_val_error_normal_cell_depth.png', dpi=300)
    plt.close()

    # PARAMETERS VS. TEST ERROR
    plt.figure(figsize=(4, 3))
    # plt.title('NAS-Bench-201 {}, num. models: {}'.format(current_time.strftime("%b %d %Y %H:%M"), len(test_errors)))
    plt.ylabel('Test error')
    plt.xlabel('Num. Parameters')
    ax = plt.gca()
    ax.set_yscale('log')
    plt.ylim(min(test_errors), max(test_errors))
    ax.set_xscale('log')
    plt.xlim(min(model_parameters), max(model_parameters))
    plt.scatter(model_parameters, test_errors, s=2, alpha=0.15, c=val_errors, norm=matplotlib.colors.LogNorm())
    cbar = plt.colorbar()
    cbar.set_label('Validation error', rotation=270)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig('analysis/plot_export/parameters_vs_test_error.png', dpi=300)
    plt.close()

    # TRAIN ERROR VS. VAL ERROR / TEST ERROR
    plt.figure(figsize=(4, 3))
    # plt.title('NAS-Bench-201 {}, num. models: {}'.format(current_time.strftime("%b %d %Y %H:%M"), len(test_errors)))
    plt.ylabel('Validation error')
    plt.xlabel('Train error')
    ax = plt.gca()
    ax.set_yscale('log')
    plt.ylim(min(val_errors), max(val_errors))
    ax.set_xscale('log')
    plt.xlim(min(train_errors), max(train_errors))
    plt.scatter(train_errors, val_errors, s=2, alpha=0.15, c=test_errors, norm=matplotlib.colors.LogNorm())
    cbar = plt.colorbar()
    cbar.set_label('Test error', rotation=270)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig('analysis/plot_export/train_error_vs_val_error.png', dpi=300)
    plt.close()

    # RUNTIME VS NUM PARAMETERS / VAL ERROR
    plt.figure(figsize=(4, 3))
    # plt.title('NAS-Bench-201 {}, num. models: {}'.format(current_time.strftime("%b %d %Y %H:%M"), len(test_errors)))
    plt.ylabel('Runtime [s]')
    plt.xlabel('Num. parameters [10e6]')
    ax = plt.gca()
    ax.set_yscale('log')
    plt.ylim(min(run_times), max(run_times))
    ax.set_xscale('log')
    plt.xlim(min(model_parameters_scaled), max(model_parameters_scaled))
    plt.scatter(model_parameters_scaled, run_times, s=1, alpha=0.2, c=val_errors, norm=matplotlib.colors.LogNorm(),
                cmap=plt.get_cmap('magma'))
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Validation Error', rotation=270, labelpad=15)
    cbar.set_ticks([0.05, 0.06, 0.1, 0.2])
    cbar.set_ticklabels(["0.05", "", "0.1", "0.2"])
    plt.yticks([2 * 10 ** 3, 10 ** 4])
    plt.xticks([0.9, 1, 2], ["", "1", "2"])
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig('analysis/plot_export/parameters_vs_runtimes.png', bbox_inches='tight', dpi=400)
    plt.close()

    # PARAMETERS VS. RUNTIME / INIT. CHANNELS
    plt.figure(figsize=(4, 3))
    # plt.title('NAS-Bench-201 {}, num. models: {}'.format(current_time.strftime("%b %d %Y %H:%M"), len(test_errors)))
    plt.ylabel('Runtime')
    plt.xlabel('Num Parameters')
    ax = plt.gca()
    ax.set_yscale('log')
    plt.ylim(min(run_times), max(run_times))
    ax.set_xscale('log')
    plt.xlim(min(model_parameters), max(model_parameters))
    plt.scatter(model_parameters, run_times, s=2, alpha=0.15, c=init_channels_model, norm=matplotlib.colors.LogNorm())
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Initial Channels', rotation=270, labelpad=15)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig('analysis/plot_export/parameters_vs_runtimes_vs_init_channels.png', dpi=300)
    plt.close()

    # VAL ERROR VS. TEST ERROR
    plt.figure(figsize=(4, 3))
    # plt.title('NAS-Bench-201 {}, num. models: {}'.format(current_time.strftime("%b %d %Y %H:%M"), len(test_errors)))
    plt.xlabel('Validation error')
    plt.ylabel('Test error')
    ax = plt.gca()
    ax.set_yscale('log')
    plt.ylim(min(test_errors), max(test_errors))
    ax.set_xscale('log')
    plt.xlim(min(val_errors), max(val_errors))
    plt.xticks([0.05, 0.06, 0.1, 0.2], ["0.05", "", "0.1", "0.2"])
    plt.yticks([0.05, 0.06, 0.1, 0.2], ["0.05", "", "0.1", "0.2"])
    plt.scatter(val_errors, test_errors, s=2, alpha=0.15)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig('analysis/plot_export/val_error_vs_test_error.png', dpi=400)
    plt.close()


if __name__ == "__main__":
    plotting()
