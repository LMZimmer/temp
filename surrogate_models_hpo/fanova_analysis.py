import itertools
import os

import click
import fanova
import fanova.visualizer
import hpbandster.core.result as hpres
import matplotlib
import numpy as np
from ConfigSpace.read_and_write import json

matplotlib.use('Agg')


def fanova_analysis(budgets, res, runs_by_budget, id2conf, bohb_logs_dir):
    """
    fANOVA analysis function.
    This plots the single marginal and pair marginal importance of the parameters in the configspace.

    :param budgets:
    :param res:
    :param runs_by_budget:
    :param id2conf:
    :param bohb_logs_dir:
    :return:
    """
    with open(os.path.join(bohb_logs_dir, 'configspace.json'), 'r') as f:
        jason_string = f.read()
    config_space = json.read(jason_string)

    for b in reversed(budgets):
        X, y, new_cs = res.get_fANOVA_data(config_space, budgets=[b])

        # Remove nan values
        nan_index = np.argwhere(np.isnan(y))
        print('budget', b, 'number nan elements', len(nan_index))
        X = np.delete(X, np.argwhere(np.isnan(y)), axis=0)
        y = np.delete(y, np.argwhere(np.isnan(y)))

        # Remove infinite values
        inf_index = np.argwhere(np.isinf(y))
        print('budget', b, 'number inf elements', len(inf_index))
        X = np.delete(X, np.argwhere(np.isinf(y)), axis=0)
        y = np.delete(y, np.argwhere(np.isinf(y)))

        f = fanova.fANOVA(X, y, new_cs)
        # Cut off the unusable configs
        f.set_cutoffs(cutoffs=(0.0, 1.0))

        dir = os.path.join(bohb_logs_dir, 'fanova', 'budget_{}'.format(b))
        os.makedirs(dir, exist_ok=True)

        vis = fanova.visualizer.Visualizer(f, new_cs, dir, y_label='Validation Error')

        print(b)

        best_run_idx = np.argsort([r.loss for r in runs_by_budget[b]])[0]
        best_run = runs_by_budget[b][best_run_idx]

        inc_conf = id2conf[best_run.config_id]['config']
        inc_conf['budget'] = best_run.budget
        inc_line_style = {'linewidth': 3, 'color': 'lightgray', 'linestyle': 'dashed'}

        for i, hp in enumerate(config_space.get_hyperparameters()):
            print(f.quantify_importance([hp.name]))
            fig = vis.plot_marginal(i, show=False, log_scale=True)  # hp.name instead of i
            fig.axvline(x=inc_conf[hp.name], **inc_line_style)
            # fig.yscale('log')
            fig.xscale('log')
            fig.title('importance %3.1f%%' % (
                    f.quantify_importance([hp.name])[(hp.name,)]['individual importance'] * 100)
                      )
            fig.tight_layout()
            fig.savefig(os.path.join(dir, '{}.png'.format(hp.name)))
            fig.close()

        for num, (hp1, hp2) in enumerate(itertools.combinations(config_space.get_hyperparameters(), 2)):
            n1, n2 = hp1.name, hp2.name
            fig = vis.plot_pairwise_marginal([n1, n2], show=False, three_d=False)
            fig.axvline(x=inc_conf[n1], **inc_line_style)
            fig.axhline(y=inc_conf[n2], **inc_line_style)
            xlims = fig.xlim()
            ylims = fig.ylim()

            fig.scatter([inc_conf[n1]], [inc_conf[n2]], color='lightgray',
                        s=800, marker='x', linewidth=5)
            fig.xlim(xlims)
            fig.ylim(ylims)

            importance = f.quantify_importance([n1, n2])[(n1, n2)]['total importance']
            fig.title("importance %3.1f%%" % (importance * 100))
            fig.title("Budget: %d epochs" % b)
            fig.tight_layout()
            fig.savefig(os.path.join(dir, 'parameter_comp_{}_{}.png'.format(hp1, hp2)))


@click.command()
@click.option("--run_name", help="Directory of Hpbandster run.", type=click.STRING)
def setup_fanova_analysis(run_name):
    bohb_logs_dir = run_name
    res = hpres.logged_results_to_HBS_result(bohb_logs_dir)

    inc_id = res.get_incumbent_id()

    id2conf = res.get_id2config_mapping()

    inc_trajectory = res.get_incumbent_trajectory()
    print(inc_trajectory)
    print(res.get_runs_by_id(inc_id))

    all_runs = list(filter(lambda r: not (r.info is None or r.loss is None),
                           res.get_all_runs()))

    budgets = res.HB_config['budgets']

    runs_by_budget = {}

    for b in budgets:
        runs_by_budget[b] = list(filter(lambda r: r.budget == b, all_runs))

    fanova_analysis(budgets=budgets, res=res, runs_by_budget=runs_by_budget, id2conf=id2conf,
                    bohb_logs_dir=bohb_logs_dir)


if __name__ == "__main__":
    setup_fanova_analysis()
