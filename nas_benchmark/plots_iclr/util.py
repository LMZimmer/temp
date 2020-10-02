import hpbandster.core.result as hpres
import numpy as np
import pandas as pd

colors = {
    'BOHB_joint': 'darkorange',
    'BOHB_nas': 'dodgerblue',
    'RE': 'crimson',
    'RS': 'darkorchid',
    'RL': 'sienna',
    'TPE': 'deepskyblue',
    'local_search': 'green',
    'LS': 'violet',
    'LS-GT': 'violet',
    'DE': 'dodgerblue',
    'DE-GT': 'dodgerblue',
    'SMAC': 'violet',
    'BANANAS': 'gold',
    'BANANAS-GT': 'gold',
    'HB': 'darkgray',
    'BANANAS-True': 'gold',
    'BANANAS-XGB': 'crimson',
    'BANANAS-GIN': 'dodgerblue',
    'True': 'gold',
    'XGB': 'crimson',
    'GIN': 'green',
    'LS-XGB': 'crimson',
    'LS-GIN': 'green',
    'RE-True': 'gold',
    'RE-XGB': 'crimson',
    'RE-GIN': 'dodgerblue',
    'local_search-True': 'gold',
    'local_search-XGB': 'crimson',
    'local_search-GIN': 'dodgerblue',
    'TPE-True': 'gold',
    'TPE-XGB': 'crimson',
    'TPE-GIN': 'dodgerblue',
    'DE-True': 'gold',
    'DE-XGB': 'crimson',
    'DE-GIN': 'dodgerblue',
    'RE true': 'crimson',
    'RE surr': 'dodgerblue',
    'DE true': 'crimson',
    'DE surr': 'dodgerblue',
    'TPE true': 'crimson',
    'TPE surr': 'dodgerblue',
    'BANANAS true': 'crimson',
    'BANANAS surr': 'dodgerblue',
    'BOHB': 'gold',
    'GT': 'dodgerblue',
    'PC_DARTS': 'green',
    'GDAS': 'red'
}

markers = {
    'BOHB_joint': '^',
    'BOHB_nas': 'v',
    'RS': 'D',
    'RE': 'o',
    'RL': 's',
    'DE': 'v',
    'SMAC': 'h',
    'BANANAS': '^',
    'HB': '>',

    'RE-True': '^',
    'RE-XGB': 'o',
    'RE-GIN': '>',

    'DE-True': '^',
    'DE-XGB': 'o',
    'DE-GIN': '>',

    'TPE-True': '^',
    'TPE-XGB': 'o',
    'TPE-GIN': '>',

    'BANANAS-True': '^',
    'BANANAS-XGB': 'o',
    'BANANAS-GIN': '>',

    'RE true': 'o',
    'RE surr': '>',
    'DE true': 'o',
    'DE surr': '>',
    'BANANAS true': 'o',
    'BANANAS surr': '>',
    'BOHB': '*',
    'TPE': '<'
}


def extract_incumbents(results, surrogate=False):
    if isinstance(results, list):
        all_runs = get_combined_runs(results)
    else:
        all_runs = results.get_all_runs()

    sorted_results = sorted(all_runs, key=lambda x:
    x.time_stamps['finished'])
    sequential_results = list()
    train_time = 0
    for x in sorted_results:
        try:
            if not surrogate:
                train_time += (x.time_stamps['finished'] -
                               x.time_stamps['started'])
            else:
                train_time += x.info['train_time']
            sequential_results.append([x.info['val_accuracy'], train_time])
        except:
            continue

    incumbent = list()
    current_incumbent = -float('inf')
    for x in sequential_results:
        if x[0] > current_incumbent:
            current_incumbent = x[0]
        incumbent.append([current_incumbent, x[1]])

    return np.array(incumbent)


def get_combined_runs(hpresult_list):
    all_runs = []
    hprun_last_timestamp = 0

    # collect runs from individual hpresults and adapt run ids and timestamps
    for hprun_id, hpresults in enumerate(hpresult_list):
        runs = hpresults.get_all_runs()
        runs = sorted(runs, key=lambda x: x.time_stamps['finished'])

        for running_id, run in enumerate(runs):
            # make run ids run from (0, 0, 1) to (0, 0, n_runs) and shift by the runs from previous hpres
            run.config_id = (0, 0, running_id + len(all_runs))
            # shift timestamps of later runs by the total runtime of combined previous runs
            for event in run.time_stamps.keys():
                run.time_stamps[event] += hprun_last_timestamp

        all_runs += runs
        hprun_last_timestamp = all_runs[-1]["time_stamps"]["finished"]

    return all_runs


def merge_and_fill_trajectories(pandas_data_frames, default_value=None):
    # merge all tracjectories keeping all time steps
    df = pd.DataFrame().join(pandas_data_frames, how='outer')

    # forward fill to make it a propper step function
    df = df.fillna(method='ffill')

    if default_value is None:
        # backward fill to replace the NaNs for the early times by
        # the performance of a random configuration
        df = df.fillna(method='bfill')
    else:
        df = df.fillna(default_value)

    return (df)


def get_trajectories_per_method(methods, suffix='', surrogate=False, append_instead_of_combining=False):
    print(methods)
    all_trajectories = {}

    for m, (paths, surrogate, append_instead_of_combining) in methods.items():
        dfs = []

        # append configs to one long trajectory
        if append_instead_of_combining:
            hp_results = []
            for i, path in enumerate(paths):
                print(path)
                true_results = hpres.logged_results_to_HBS_result(path)
                hp_results.append(true_results)
            true_inc = extract_incumbents(hp_results, surrogate=surrogate)
            error = 1 - true_inc[:, 0] / 100
            times = true_inc[:, 1]
            df = pd.DataFrame({str(0): error}, index=times)
            dfs.append(df)

        # average over trajectories
        else:
            for i, path in enumerate(paths):
                print(path)
                true_results = hpres.logged_results_to_HBS_result(path)
                true_inc = extract_incumbents(true_results, surrogate=surrogate)
                error = 1 - true_inc[:, 0] / 100
                times = true_inc[:, 1]
                df = pd.DataFrame({str(i): error}, index=times)
                dfs.append(df)

        df_true = merge_and_fill_trajectories(dfs, default_value=None)

        all_trajectories[m + suffix] = {
            'time_stamps': np.array(df_true.index),
            'errors': np.array(df_true.T)
        }

    return all_trajectories


def get_trajectories(true_paths, surrogate_paths, methods=['BANANAS'],
                     surrogate='xgb'):
    print(true_paths)
    print(surrogate_paths)
    all_trajectories = {}

    for m in methods:
        dfs = []
        for i, true_path in enumerate(true_paths):
            print(true_path)
            true_results = hpres.logged_results_to_HBS_result(true_path)
            true_inc = extract_incumbents(true_results, surrogate=False)
            error = 100 - true_inc[:, 0]
            times = true_inc[:, 1]
            df = pd.DataFrame({str(i): error}, index=times)
            dfs.append(df)

        df_true = merge_and_fill_trajectories(dfs, default_value=None)

        dfs = []
        for i, surr_path in enumerate(surrogate_paths):
            try:
                print(surr_path)
                surr_results = hpres.logged_results_to_HBS_result(surr_path)
                surr_inc = extract_incumbents(surr_results, surrogate=True)
                error = 100 - surr_inc[:, 0]
                times = surr_inc[:, 1]
                df = pd.DataFrame({str(i): error}, index=times)
                dfs.append(df)
            except Exception as e:
                print('Could not  read:', surr_path)

        df_surr = merge_and_fill_trajectories(dfs, default_value=None)

        all_trajectories[m + ' true'] = {
            'time_stamps': np.array(df_true.index),
            'errors': np.array(df_true.T)
        }
        all_trajectories[m + ' surr'] = {
            'time_stamps': np.array(df_surr.index),
            'errors': np.array(df_surr.T)
        }

    return all_trajectories


def plot_losses(fig, ax, incumbent_trajectories,
                incumbent=None, show=True, linewidth=3, marker_size=10,
                xscale='log', xlabel='wall clock time [s]', yscale='log',
                ylabel=None, legend_loc='best', xlim=None, ylim=None,
                plot_mean=True, labels={}, markers=markers, colors=colors,
                figsize=(16, 9)):
    max_len = float('inf')
    for m, tr in incumbent_trajectories.items():
        max_time = np.copy(tr['time_stamps'])[-1]
        if max_time < max_len:
            max_len = max_time

    for m, tr in incumbent_trajectories.items():
        trajectory = np.copy(tr['errors'])
        if (trajectory.shape[0] == 0): continue

        # sem  =  np.sqrt(trajectory.var(axis=0, ddof=1)/tr['errors'].shape[0])
        sem = trajectory.std(axis=0, ddof=1)
        if plot_mean:
            mean = trajectory.mean(axis=0)
        else:
            mean = np.median(trajectory, axis=0)
            sem *= 1.253

        ax.fill_between(tr['time_stamps'], mean - 2 * sem, mean + 2 * sem,
                        color=colors[m], alpha=0.2)

        ax.plot(tr['time_stamps'], mean,
                label=labels.get(m, m), color=colors.get(m, None), linewidth=linewidth,
                markersize=marker_size, markevery=(0.1, 0.1))

    return max_len

def fill_plot_one_shot(trajectories, fig, ax, title, first=False, marker_size=10,
                       linewidth=3, xscale='linear', yscale='linear'):

    for m, tr in trajectories.items():

        mean = np.mean(tr, axis=0)
        sem = np.std(tr, axis=0)

        epochs = list(range(1,len(mean)+1))
        
        ax.fill_between(epochs, mean - 2 * sem, mean + 2 * sem,
                color=colors[m], alpha=0.2)

        ax.plot(epochs, mean,
                label=m, color=colors.get(m, None), linewidth=linewidth,
                markersize=marker_size, markevery=(0.1, 0.1))

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    #ax.set_ylim([4.7 / 100, 25/100]) #s1
    #ax.set_ylim([4.7 / 100, 12/100]) #s2, s3
    ax.set_ylim([4.5 / 100, 10/100]) #original
    if first:
        ax.legend()
        ax.set_ylabel('Validation error')
    ax.set_xlabel('Epochs')
    ax.set_title(title)
    ax.grid(True, which="both", ls="-", alpha=0.3)

