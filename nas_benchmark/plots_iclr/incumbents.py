import os

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from nas_benchmark.plots_iclr.util import get_trajectories_per_method, plot_losses

matplotlib.use("Agg")
sns.set_style('whitegrid')
matplotlib.rcParams.update({'font.size': 14})

method_dict = lambda method, dict: {
    'True': (methods_gt[method][0], False, False),
    'GIN': (get_surrogate_paths_for_method('gnn_gin', dict[method], method), True, False),
    'XGB': (get_surrogate_paths_for_method('XGB', dict[method], method), True, False)
}

surr_directory_map = {
    'DE': '/data/aad/image_datasets/nasbench_301/paper_paper_iclr_all_data_final',
    'BANANAS': '/data/aad/image_datasets/nasbench_301/paper_paper_iclr_all_data_final',
    'RS': '/data/aad/image_datasets/nasbench_301/paper_paper_iclr_all_data_final',
    'TPE': '/data/aad/image_datasets/nasbench_301/paper_paper_iclr_all_data_final',
    'RE': '/data/aad/image_datasets/nasbench_301/paper_paper_iclr_all_data_final',
    # 'local_search': '/data/aad/image_datasets/nasbench_301/paper_paper_iclr_all_data_final'
}

surr_directory_map_loo = {
    'DE': '/home/user/HpBandSter/eval/paper_paper_iclr_looo_final_lucas',
    'BANANAS': '/data/aad/image_datasets/nasbench_301/paper_paper_iclr_looo_final_3',
    'RE': '/home/user/HpBandSter/eval/paper_paper_iclr_looo_final_lucas',
    'local_search': '/data/aad/image_datasets/nasbench_301/paper_paper_iclr_all_data_final'
}

surr_directory_map_ablation = {
    'BANANAS': '/data/aad/image_datasets/nasbench_301/paper_ablation/',
    'RS': '/data/aad/image_datasets/nasbench_301/paper_ablation_2/'
}

method_name_map = {
    'xgb': 'XGB',
    'gnn_gin': 'GIN'
}

methods_gt = {
    'DE': (['/home/user/projects/HpBandSter_test/eval/de/darts_fidelity_bosch_32_8_100_96_673'], False, False),
    'RE': (['/home/user/projects/HpBandSter/eval/re/darts_fidelity/nb_301_32_8_100_96_0'], False, False),
    'TPE': (['/home/user/projects/HpBandSter/eval/tpe/darts_fidelity/nb_301_1_32_8_100_96_1010'], False, False),
    'BANANAS': (
        ['/home/user/projects/HpBandSter_bananas/eval/bananas/darts_fidelity/nb_301_test_32_8_100_96_4'], False,
        False),
    'RS': (['/home/user/projects/HpBandSter/eval/rs/darts_fidelity/nb_301_rtx_2080_32_8_100_96_1021',
            '/home/user/projects/HpBandSter/eval/rs/darts_fidelity/nb_301_rtx_2080_32_8_100_96_1012',
            '/home/user/projects/HpBandSter/eval/rs/darts_fidelity/nb_301_rtx_2080_32_8_100_96_1020',
            '/home/user/HpBandSter/eval/rs/darts_fidelity/nb_301_32_8_100_96_904'
            ], False, True),
    'local_search': (
        ['/home/user/projects/HpBandSter_bananas/eval/local_search/darts_fidelity_2/_32_8_100_96_1000'], False, False)
}


def get_surrogate_paths_for_method(surrogate, surr_directory, method):
    surrogates_dir = os.path.join(surr_directory, method.lower())
    surrogate_dirs = list(filter(lambda x: x.startswith(surrogate.lower()), os.listdir(surrogates_dir)))

    surrogate_paths = [os.path.join(surrogates_dir, d) for d in surrogate_dirs]
    return surrogate_paths


def fill_plot(methods, fig, ax, title, first=False):
    trajectories = get_trajectories_per_method(methods, surrogate=True)
    max_time = plot_losses(fig, ax, trajectories, plot_mean=True)

    ax.set_xscale('log')
    ax.set_yscale('log')
    # ax.ticklabel_format(axis='y', style='plain')
    ax.set_ylim([4.7 / 100, 9 / 100])
    if first:
        ax.legend()
        ax.set_ylabel('Best validation error achieved')
        ax.set_xlabel('Wallclock Time [s]')
    else:
        ax.set_xlabel('Simulated Wallclock Time [s]')
    ax.set_title(title)
    ax.grid(True, which="both", ls="-", alpha=0.3)


def plot_all_data_results_common_y():
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, nrows=1, sharey=True, figsize=(13, 4))

    fill_plot(methods_gt, fig, ax1, title="True Benchmark", first=True)

    methods = {
        method: (get_surrogate_paths_for_method('gnn_gin', surr_directory_map[method], method), True, False) for method
        in surr_directory_map.keys()
    }
    fill_plot(methods, fig, ax2, title="{} Surrogate Benchmark".format(method_name_map['gnn_gin']))

    methods = {
        method: (get_surrogate_paths_for_method('xgb', surr_directory_map[method], method), True, False) for method in
        surr_directory_map.keys()
    }
    fill_plot(methods, fig, ax3, title="{} Surrogate Benchmark".format(method_name_map['xgb']), first=False)

    plt.tight_layout()
    os.makedirs('./plot_results', exist_ok=True)
    fig_name = './plot_results' + '/benchmark_all_data.png'
    plt.savefig(fig_name, dpi=300)
    plt.close()


def fill_plot_loo(methods, fig, ax, title, first=False):
    trajectories = get_trajectories_per_method(methods, surrogate=True)
    max_time = plot_losses(fig, ax, trajectories, plot_mean=True)

    ax.set_xscale('log')
    ax.set_yscale('log')
    # ax.ticklabel_format(axis='y', style='plain')
    ax.set_ylim([4.7 / 100, 9 / 100])
    ax.set_xlim([0, 4*10**7])
    if first:
        ax.legend()
        ax.set_xlabel('(Simulated) Wallclock Time [s]')
        ax.set_ylabel('Best validation error achieved')
    else:
        # ax.legend()
        ax.set_xlabel('(Simulated) Wallclock Time [s]')
    ax.set_title(title)
    ax.grid(True, which="both", ls="-", alpha=0.2)


def fill_plot_local_search(methods, fig, ax, title, first=False):
    trajectories = get_trajectories_per_method(methods, surrogate=True)
    max_time = plot_losses(fig, ax, trajectories, plot_mean=True)

    ax.set_xscale('log')
    ax.set_yscale('log')
    # ax.ticklabel_format(axis='y', style='plain')
    ax.set_ylim([4.7 / 100, 9 / 100])
    ax.set_xlim([0, 4*10**7])
    if first:
        ax.legend(fontsize='small')
        ax.set_xlabel('(Simulated) Wallclock Time [s]')
        ax.set_ylabel('Best validation error achieved')
    else:
        # ax.legend()
        ax.set_xlabel('(Simulated) Wallclock Time [s]')
    ax.set_title(title)
    ax.grid(True, which="both", ls="-", alpha=0.2)


def plot_leave_one_out_common_y():
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, nrows=1, sharey=True, figsize=(13, 4))

    fill_plot_loo(method_dict('BANANAS', surr_directory_map_loo), fig, ax1, title="BANANAS", first=True)
    fill_plot_loo(method_dict('RE', surr_directory_map_loo), fig, ax2, title="RE", first=False)
    fill_plot_loo(method_dict('DE', surr_directory_map_loo), fig, ax3, title="DE", first=False)

    plt.tight_layout()
    os.makedirs('./plot_results', exist_ok=True)
    fig_name = './plot_results' + '/benchmark_looo.png'
    plt.savefig(fig_name, dpi=300)
    plt.close()


def plot_local_search():
    fig, (ax1) = plt.subplots(ncols=1, nrows=1, sharey=True, figsize=(4.7, 4))
    methods = method_dict('local_search', surr_directory_map_loo)
    methods['LS-GT'] = methods.pop('True')
    methods['LS-GIN'] = methods.pop('GIN')
    methods['LS-XGB'] = methods.pop('XGB')
    for opt_gt in ['BANANAS', 'DE']:
        methods[opt_gt + '-GT'] = methods_gt[opt_gt]

    fill_plot_local_search(methods, fig, ax1, title="Local Search", first=True)

    plt.tight_layout()
    os.makedirs('./plot_results', exist_ok=True)
    fig_name = './plot_results' + '/benchmark_looo_local_search.png'
    plt.savefig(fig_name, dpi=300)
    plt.close()


def plot_ablation_study():
    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, sharey=True, figsize=(9, 4))
    fill_plot(method_dict('BANANAS', surr_directory_map_ablation), fig, ax1, title="BANANAS", first=True)
    fill_plot(method_dict('RS', surr_directory_map_ablation), fig, ax2, title="RS", first=False)

    plt.tight_layout()
    os.makedirs('./plot_results', exist_ok=True)
    fig_name = './plot_results' + '/benchmark_ablation.png'
    plt.savefig(fig_name, dpi=300)
    plt.close()


if __name__ == '__main__':
    # plot_ablation_study()
    # plot_local_search()
    # plot_leave_one_out_common_y()
    plot_all_data_results_common_y()
