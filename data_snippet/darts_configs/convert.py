import json
import yaml
import os
import re

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from autoPyTorch import AutoNetImageClassification
from genotypes import Genotype

from IPython import embed

def get_sampling_space():
    sampling_space = dict()
    sampling_space["batch_loss_computation_techniques"] = ['mixup']
    sampling_space["networks"] = ['darts']
    sampling_space["optimizer"] = ['sgd']
    sampling_space["lr_scheduler"] = ['cosine_annealing']
    sampling_space["loss_modules"] = ['cross_entropy']
    return sampling_space

def read_genotype_from_yaml(filename='configs.yaml'):
    with open(filename) as f:
        archs = yaml.load(f)
    return archs


def convert_genotype_to_config(arch):
    base_string = 'NetworkSelectorDatasetInfo:darts:'
    config = {}

    for cell_type in ['normal', 'reduce']:
        cell = eval('arch.'+cell_type)

        start = 0
        n = 2
        for node_idx in range(4):
            end = start + n
            ops = cell[2*node_idx: 2*node_idx+2]

            # get edge idx
            edges = {base_string+'edge_'+cell_type+'_'+str(start+i): op for
                     op, i in ops}
            config.update(edges)

            if node_idx != 0:
                # get node idx
                input_nodes = sorted(list(map(lambda x: x[1], ops)))
                input_nodes_idx = '_'.join([str(i) for i in input_nodes])
                config.update({base_string+'inputs_node_'+cell_type+'_'+str(node_idx+2):
                               input_nodes_idx})

            start = end
            n += 1

    return config


if __name__=="__main__":

    sampling_space = get_sampling_space()
    autonet = AutoNetImageClassification(**sampling_space)

    fixed_hyperpars = {
        "CreateImageDataLoader:batch_size": 96,
        "ImageAugmentation:augment": "True",
        "ImageAugmentation:cutout": "True",
        "ImageAugmentation:cutout_holes": 1,
        "ImageAugmentation:cutout_length": 16,
        "ImageAugmentation:fastautoaugment": "False",
        "LossModuleSelectorIndices:loss_module": "cross_entropy",
        "NetworkSelectorDatasetInfo:darts:auxiliary": "True",
        "NetworkSelectorDatasetInfo:darts:drop_path_prob": 0.2,
        "NetworkSelectorDatasetInfo:network": "darts",
        "OptimizerSelector:optimizer": "sgd",
        "OptimizerSelector:sgd:learning_rate": 0.025,
        "OptimizerSelector:sgd:momentum": 0.9,
        "OptimizerSelector:sgd:weight_decay": 0.0003,
        "SimpleLearningrateSchedulerSelector:cosine_annealing:T_max": 600,
        "SimpleLearningrateSchedulerSelector:cosine_annealing:eta_min": 0,
        "SimpleLearningrateSchedulerSelector:lr_scheduler": "cosine_annealing",
        "SimpleTrainNode:batch_loss_computation_technique": "mixup",
        "SimpleTrainNode:mixup:alpha": 0.0,
        "NetworkSelectorDatasetInfo:darts:init_channels": 36,
        "NetworkSelectorDatasetInfo:darts:layers": 20
        }

    cs = autonet.get_hyperparameter_search_space()
    hyperparameter_config = cs.get_default_configuration().get_dictionary()

    # convert string to bool
    for key in hyperparameter_config.keys():
        if hyperparameter_config[key]==True:
            hyperparameter_config[key]="True"
        elif hyperparameter_config[key]==False:
            hyperparameter_config[key]="False"

    # set fixed parameters
    combined_config = {**hyperparameter_config, **fixed_hyperpars}

    # remove all architectural hyperparameters in order to set them again
    combined_config = {
        k: v for (k, v) in combined_config.items() if not re.search('.normal|reduce.', k)
    }

    darts_archs = read_genotype_from_yaml()

    if not os.path.exists('configs'):
        os.makedirs('configs')

    for idx in darts_archs:
        for i, arch in enumerate(darts_archs[idx]):
            arch = eval(arch)
            config = convert_genotype_to_config(arch)
            config.update(combined_config.copy())

            # dump
            with open("configs/config_%d_%d"%(idx, i) + ".json", "w") as f:
                json.dump(config, f)

