import json
from collections import defaultdict

from tqdm import tqdm

from surrogate_models.utils import ConfigLoader


"""
SCRIPT TO CREATE THE GROUNDTRUTH DATA FOR THE NORMAL/REDUCTION CELL TOPOLOGY ANALYSIS.
"""

def get_graph_topologies():
    config_space = ConfigLoader('configspace.json').config_space

    # Sample architectures from search space
    sample_archs = [config_space.sample_configuration() for i in range(100000)]

    # Extract the normal cell topologies
    normal_cell_topologies = defaultdict(list)
    for arch in tqdm(sample_archs):
        normal_cell_topology = {
            'NetworkSelectorDatasetInfo:darts:inputs_node_normal_{}'.format(idx): arch[
                'NetworkSelectorDatasetInfo:darts:inputs_node_normal_{}'.format(idx)] for idx in range(3, 6)
        }
        arch_hash = hash(frozenset(normal_cell_topology.items()))
        if len(normal_cell_topologies[arch_hash]) < 10:
            normal_cell_topologies[arch_hash].append(arch.get_dictionary())

    assert len(normal_cell_topologies) == 180, 'Not all connectivity patterns were sampled.'
    assert all([len(archs) == 10 for normal_cell, archs in
                normal_cell_topologies.items()]), 'The number of configs for each normal wasnt fulfilled'
    json.dump(normal_cell_topologies, open('normal_cell_topologies.json', 'w'))


def replace_normal_cell():
    normal_cell_topologies = json.load(open('normal_cell_topologies.json', 'r'))
    normal_cell_topologies_new = defaultdict(list)

    for normal_cell_topology, archs in normal_cell_topologies.items():
        for arch in archs:
            # Replace topology in reduction cell with normal cell's
            for inter_node in range(3, 6):
                arch['NetworkSelectorDatasetInfo:darts:inputs_node_reduce_{}'.format(inter_node)] = \
                    arch['NetworkSelectorDatasetInfo:darts:inputs_node_normal_{}'.format(inter_node)]
            # Replace operations in reduction cell with normal cell's
            for op_idx in range(14):
                # First remove the existing operation in reduction cell if it exists.
                arch.pop('NetworkSelectorDatasetInfo:darts:edge_reduce_{}'.format(op_idx), None)
                if 'NetworkSelectorDatasetInfo:darts:edge_normal_{}'.format(op_idx) in arch:
                    arch['NetworkSelectorDatasetInfo:darts:edge_reduce_{}'.format(op_idx)] = \
                        arch['NetworkSelectorDatasetInfo:darts:edge_normal_{}'.format(op_idx)]
        normal_cell_topologies_new[normal_cell_topology].append(archs)

    json.dump(normal_cell_topologies, open('normal_cell_topologies_replicated_normal_and_reduction_cell.json', 'w'))


if __name__ == "__main__":
    get_graph_topologies()
    replace_normal_cell()
