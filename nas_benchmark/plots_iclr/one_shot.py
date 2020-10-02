import os
import json
from pathlib import Path

import click
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from IPython import embed

from surrogate_models import utils
from surrogate_models.ensemble import Ensemble
from nas_benchmark.plots_iclr.util import fill_plot_one_shot

matplotlib.use("Agg")
sns.set_style('whitegrid')
matplotlib.rcParams.update({'font.size': 14})

fixed_hyperparameters = {
        "CreateImageDataLoader:batch_size": 96,
        "ImageAugmentation:augment": "True",
        "ImageAugmentation:cutout": "True",
        "ImageAugmentation:cutout_holes": 1,
        "ImageAugmentation:cutout_length": 16,
        "ImageAugmentation:autoaugment": "False",
        "ImageAugmentation:fastautoaugment": "False",
        "LossModuleSelectorIndices:loss_module": "cross_entropy",
        "NetworkSelectorDatasetInfo:darts:auxiliary": "True",
        "NetworkSelectorDatasetInfo:darts:drop_path_prob": 0.2,
        "NetworkSelectorDatasetInfo:network": "darts",
        "OptimizerSelector:optimizer": "sgd",
        "OptimizerSelector:sgd:learning_rate": 0.025,
        "OptimizerSelector:sgd:momentum": 0.9,
        "OptimizerSelector:sgd:weight_decay": 0.0003,
        "SimpleLearningrateSchedulerSelector:cosine_annealing:T_max": 100,
        "SimpleLearningrateSchedulerSelector:cosine_annealing:eta_min": 1e-8,
        "SimpleLearningrateSchedulerSelector:lr_scheduler": "cosine_annealing",
        "SimpleTrainNode:batch_loss_computation_technique": "mixup",
        "SimpleTrainNode:mixup:alpha": 0.2,
        "NetworkSelectorDatasetInfo:darts:init_channels": 32,
        "NetworkSelectorDatasetInfo:darts:layers": 8
        }

one_shot_gt = {
    'DARTS': '/home/user/MultiAutoPyTorch_LZ/logs/darts_proxy/',
    'PC_DARTS': '/home/user/MultiAutoPyTorch_LZ/logs/pcdarts_proxy',
    'GDAS': '/home/user/MultiAutoPyTorch_LZ/logs/gdas_proxy',
    'DARTS_SUB': '/home/user/MultiAutoPyTorch_LZ/logs/darts_proxy/'
}

trajectory_lengths = {
        'DARTS_SUB': 50,
        'PC_DARTS': 50,
        'GDAS': 50,
        'DARTS': 50
        }

def chunk_list(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def load_surrogate_ensemble(ensemble_parent_directory):

    ensemble_member_dirs = [os.path.dirname(filename) for filename in Path(ensemble_parent_directory).rglob('*surrogate_model.model')]
    data_config = json.load(open(os.path.join(ensemble_member_dirs[0], 'data_config.json'), 'r'))
    model_config = json.load(open(os.path.join(ensemble_member_dirs[0], 'model_config.json'), 'r'))

    surrogate_model = Ensemble(member_model_name=model_config['model'],
                               data_root='None',
                               log_dir=ensemble_parent_directory,
                               starting_seed=data_config["seed"],
                               model_config=model_config,
                               data_config=data_config,
                               ensemble_size=len(ensemble_member_dirs),
                               init_ensemble=False)

    surrogate_model.load(model_paths=ensemble_member_dirs)
    return surrogate_model

def load_ordered_results(path):
    rundirs = [p for p in os.listdir(path) if p.startswith("run_")]
    rundirs.sort(key=lambda x: int(x.split("_")[-1]))
    rundirs = ["run_"+str(ind) for ind in range(1,int(rundirs[-1].split("_")[-1])+1)] # fill failed runs

    ordered_results = []
    for rundir in rundirs:
        result_dir = os.path.join(path, rundir, "final_output.json")
        if os.path.exists(result_dir):
            result_dict = json.load(open(result_dir, "r"))
            ordered_results.append(result_dict)
        else:
            ordered_results.append(ordered_results[-1])
    return ordered_results

def load_groundtruth_trajectories(darts_indices_min=1950, darts_indices_max=2200):
    trajectories = dict()
    configs = dict()

    for method, path in one_shot_gt.items():

        trajectory_length = trajectory_lengths[method]

        all_results_ordered = load_ordered_results(path)
        if method=="DARTS_SUB":
            # 1600-1700: s1
            # 1700-1950: s2
            # 1950-2200: s3
            all_results_ordered = all_results_ordered[darts_indices_min:darts_indices_max] # DARTS trajectories start at 1601
        if method=="DARTS":
            all_results_ordered = all_results_ordered[2200:2401] # DARTS original trajectories start at 2001
        all_configs_ordered = [{**result["optimized_hyperparamater_config"], **fixed_hyperparameters} for result in all_results_ordered]
        all_groundtruths_ordered = [(100-result["info"][0]["val_accuracy"])/100 for result in all_results_ordered]
        trajectory_configs = list(chunk_list(all_configs_ordered, trajectory_length))
        groundtruth_trajectories = list(chunk_list(all_groundtruths_ordered, trajectory_length)) # trajectories all have teh same num. epochs and results are ordered
        print("==> Found %i trajectories for %s" %(len(groundtruth_trajectories), method))

        configs[method] = trajectory_configs
        trajectories[method] = groundtruth_trajectories

    return configs, trajectories

def simulate_trajectories(surrogate_model, trajectory_configs):
    surrogate_trajectories = dict()

    for method, config_trajectories in trajectory_configs.items():
        all_trajectories = []
        for single_config_trajectory in config_trajectories:
            predictions = [(100-surrogate_model.query(config_dict))/100 for config_dict in single_config_trajectory]
            all_trajectories.append(predictions)

        surrogate_trajectories[method] = all_trajectories

    return surrogate_trajectories

def simulate_loo_trajectories(surrogate_models, trajectory_configs):
    surrogate_trajectories = dict()

    for method, config_trajectories in trajectory_configs.items():
        all_trajectories = []
        for single_config_trajectory in config_trajectories:
            surrogate_model = surrogate_models[method]
            predictions = [(100-surrogate_model.query(config_dict))/100 for config_dict in single_config_trajectory]
            all_trajectories.append(predictions)

        surrogate_trajectories[method] = all_trajectories

    return surrogate_trajectories

    
@click.command()
@click.option('--ensemble_parent_dir_gnn', type=click.STRING, help='Directory containing the ensemble members', default='none')
@click.option('--ensemble_parent_dir_xgb', type=click.STRING, help='Directory containing the ensemble members', default='none')
@click.option('--ensemble_parent_dir_loo_darts_gnn', type=click.STRING, help='Directory containing the ensemble members', default='none')
@click.option('--ensemble_parent_dir_loo_darts_xgb', type=click.STRING, help='Directory containing the ensemble members', default='none')
@click.option('--ensemble_parent_dir_loo_gdas_gnn', type=click.STRING, help='Directory containing the ensemble members', default='none')
@click.option('--ensemble_parent_dir_loo_gdas_xgb', type=click.STRING, help='Directory containing the ensemble members', default='none')
@click.option('--ensemble_parent_dir_loo_pcdarts_gnn', type=click.STRING, help='Directory containing the ensemble members', default='none')
@click.option('--ensemble_parent_dir_loo_pcdarts_xgb', type=click.STRING, help='Directory containing the ensemble members', default='none')
@click.option('--plot_identifier', type=click.STRING, help='ID to append to the plotname, e.g. loo_s1 or all_data_s1')
@click.option('--plot_single', is_flag=True)
def plot_one_shot_trajectories(ensemble_parent_dir_gnn, ensemble_parent_dir_xgb, ensemble_parent_dir_loo_darts_gnn, ensemble_parent_dir_loo_darts_xgb,
                               ensemble_parent_dir_loo_gdas_gnn, ensemble_parent_dir_loo_gdas_xgb, ensemble_parent_dir_loo_pcdarts_gnn,
                               ensemble_parent_dir_loo_pcdarts_xgb, plot_identifier, plot_single):

    print("==> Loading surrogates...")
    if ensemble_parent_dir_gnn != 'none' and ensemble_parent_dir_xgb != 'none':
        surrogate_model_gnn = load_surrogate_ensemble(ensemble_parent_dir_gnn)
        surrogate_model_xgb = load_surrogate_ensemble(ensemble_parent_dir_xgb)
    else:
        surrogate_model_gnn_darts = load_surrogate_ensemble(ensemble_parent_dir_loo_darts_gnn)
        surrogate_model_xgb_darts = load_surrogate_ensemble(ensemble_parent_dir_loo_darts_xgb)

        surrogate_model_gnn_gdas = load_surrogate_ensemble(ensemble_parent_dir_loo_gdas_gnn)
        surrogate_model_xgb_gdas = load_surrogate_ensemble(ensemble_parent_dir_loo_gdas_xgb)

        surrogate_model_gnn_pcdarts = load_surrogate_ensemble(ensemble_parent_dir_loo_pcdarts_gnn)
        surrogate_model_xgb_pcdarts = load_surrogate_ensemble(ensemble_parent_dir_loo_pcdarts_xgb)

        loo_surrogate_models_gnn = {
                "DARTS": surrogate_model_gnn_darts,
                "DARTS_SUB": surrogate_model_gnn_darts,
                "GDAS": surrogate_model_gnn_gdas,
                "PC_DARTS": surrogate_model_gnn_pcdarts
                }

        loo_surrogate_models_xgb = {
                "DARTS": surrogate_model_xgb_darts,
                "DARTS_SUB": surrogate_model_gnn_darts,
                "GDAS": surrogate_model_xgb_gdas,
                "PC_DARTS": surrogate_model_xgb_pcdarts
                }


    print("==> Loading groundtruth...")
    trajectory_configs, groundtruth_trajectories = load_groundtruth_trajectories()

    print("==> Querying surrogates...")
    if ensemble_parent_dir_gnn != 'none' and ensemble_parent_dir_xgb != 'none':
        surrogate_trajectories_gnn = simulate_trajectories(surrogate_model=surrogate_model_gnn, trajectory_configs=trajectory_configs)
        surrogate_trajectories_xgb = simulate_trajectories(surrogate_model=surrogate_model_xgb, trajectory_configs=trajectory_configs)
    else:
        surrogate_trajectories_gnn = simulate_loo_trajectories(surrogate_models=loo_surrogate_models_gnn,
                                                               trajectory_configs=trajectory_configs)
        surrogate_trajectories_xgb = simulate_loo_trajectories(surrogate_models=loo_surrogate_models_xgb,
                                                               trajectory_configs=trajectory_configs)
    
    print("==> Plotting...")
    if not plot_single:
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, nrows=1, sharey=True, figsize=(13, 4))
        fill_plot_one_shot(groundtruth_trajectories, fig, ax1, title="True Benchmark", first=True)
        fill_plot_one_shot(surrogate_trajectories_gnn, fig, ax2, title="{} Surrogate Benchmark".format('GIN'), first=False)
        fill_plot_one_shot(surrogate_trajectories_xgb, fig, ax3, title="{} Surrogate Benchmark".format('XGB'), first=False)
    
    else:
        trajectories_all_data = {
                "GT": groundtruth_trajectories["DARTS_SUB"],
                "GIN": surrogate_trajectories_gnn["DARTS_SUB"],
                "XGB": surrogate_trajectories_xgb["DARTS_SUB"],
                }

        fig, ax1 = plt.subplots(ncols=1, nrows=1, sharey=True, figsize=(4.5, 4))
        fill_plot_one_shot(trajectories_all_data, fig, ax1, title="All data", first=True)
        #fill_plot_one_shot(trajectories_all_data, fig, ax1, title="LOTO", first=False)
        #fill_plot_one_shot(trajectories_all_data, fig, ax1, title="LOOO", first=False)


    plt.tight_layout()
    os.makedirs('./plot_results', exist_ok=True)
    fig_name = './plot_results' + '/benchmark_one_shot_' + plot_identifier + '.png'
    plt.savefig(fig_name, dpi=300)
    plt.close()
    
    print("==> Done")

if __name__ == '__main__':
    plot_one_shot_trajectories()
