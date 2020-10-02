import glob
import json
import os as os

import click
import numpy as np

from surrogate_models import utils

from IPython import embed

@click.command()
@click.option('--rs_data', type=click.STRING, help='Path to random search data directory', default="/home/user/projects/nasbench_201_2/analysis/nb_301_v12/rs")
@click.option('--output_dir', type=click.STRING, help='Path to save configs to.', default="/home/user/configs/refit/lowpar_training_data/.")
def create_lowpar_configs(rs_data, output_dir):

    NUM_CONFIGS_PER_STEP = 112 + 1

    os.makedirs(output_dir, exist_ok=True)
    config_paths = glob.glob(os.path.join(rs_data, '*.json'))
    config_loader = utils.ConfigLoader('configspace.json')

    # Take all datapoints in a diagonal fidelity and transform them to the other fidelities.
    # Iterate through the parameter types
    total_num_configs = 0
    ratio_skip_connection_in_cell_dict = {'max_pool_3x3': {}, 'avg_pool_3x3': {}, 'skip_connect': {}}
    for parameter_free_op in ratio_skip_connection_in_cell_dict.keys():
        config_loader.parameter_free_op_increase_type = parameter_free_op
        # Progressively increase the ratio of the selected parameter free operation
        for ratio_parameter_free_op_in_cell in np.arange(0.375, 1.1, 2 / 8):
            config_loader.ratio_parameter_free_op_in_cell = ratio_parameter_free_op_in_cell
            for path in config_paths[:NUM_CONFIGS_PER_STEP]:
                config_space_instance, _, _, _ = config_loader[path]
                config_space_instance = config_space_instance.get_dictionary()

                total_num_configs+=1
                save_dir = os.path.join(output_dir, "config_"+str(total_num_configs)+".json")
                json.dump(config_space_instance, open(save_dir, "w"))

    print("==> Done creating %i configs" %total_num_configs)

if __name__ == "__main__":
    create_lowpar_configs()
