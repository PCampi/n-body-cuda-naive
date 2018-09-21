"""Postprocessing"""

import math
from typing import List, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib.pyplot as plt
import seaborn as sns


def plot_complete(result,
                  n_iter,
                  colors: List[Tuple[str, str]],
                  logx=False,
                  logy=False,
                  asympt=None,
                  asympt_color=None,
                  asympt_label=None):
    """
    Create the time vs threads plot.

    Parameters
    ----------
    result:
        the result of a series of runs. Must contain keys:
            - bodies
            - threads
            - lower
            - mean
            - upper
    
    n_iter: int
        number of iterations for the result
    
    colors: List[Tuple[str, str]]
        a list of tuples, of length >= len(n_bodies)
        defines the colours of the plot, where:
            - colours[i][0] is the marker and line colour
            - colours[i][1] is the CI fill colour
    
    logx: bool, default False
        if the x scale is logarithmic
    
    logy: bool, default False
        if the y scale is logarithmic
    
    asympt: np.array or list
        the asymptotic behaviour of the data. If None, it will not be plotted.
        Must have len(asympt) = len(threads)
    
    asympt_color: str
        color of the asymptotic behaviour
    
    asympt_label: str
        label of the asymptotic behaviour
    """
    bodies = result['bodies']
    threads = result['threads']
    mean = result['mean']
    lower = result['lower']
    upper = result['upper']

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 12))

    for i, N in enumerate(bodies):
        ax.plot(
            threads[i],
            mean[i],
            marker='o',
            color=colors[i][0],
            label=f"{N} bodies")
        ax.fill_between(threads[i], lower[i], upper[i], color=colors[i][1])

    if (asympt is not None) and asympt_color and asympt_label:
        ax.plot(threads[0], asympt, linestyle=':', color=asympt_color, label=asympt_label)

    ax.legend()
    plt.title(f"Wall time vs number of threads on CPU for {n_iter} iterations")
    plt.xlabel("Thread number")
    plt.ylabel("Wall time (s)")

    sns.despine()

    if logx:
        plt.xscale('log')
    if logy:
        plt.yscale('log')

    return fig, ax


def get_mean_ci(data, confidence=0.95):
    """
    Get mean and confidence interval of the provided data.

    Parameters
    ----------
    data: np.ndarray of shape (n,)
        the data
    
    confidence: float in range[0.0, 1.0]
        the confidence level of the test
    
    Returns
    -------
    mean, lower, upper: Tuple[float, float, float]
        the mean of the data, the upper and lower bounds of the CI
        mean = sum(data) / len(data)
        
        CI is computed assuming unknown variance, using the t-student
        distribution with a confidence level of 95% --> 0.95
        mean +- t((1 - 0.95) / 2, N - 1) * (std_dev / sqrt(N))
    """
    N = len(data)
    if N == 1:
        raise ValueError("Data only has a single point!")

    mean = np.mean(data)
    stderr = np.std(data)
    dof = N - 1

    t_low, t_up = t.interval(confidence, dof)
    lower, upper = mean + t_low * (stderr / math.sqrt(N)), mean + t_up * (
        stderr / math.sqrt(N))

    return mean, lower, upper


def compute_results(filename):
    """
    Main function.
    """
    data = pd.read_csv(filename, sep=',')
    data.loc[:, 'elapsed'] = data['end'] - data['start']

    # 1. create threads - vs - times data to plot
    unique_body_num = np.unique(data['nbodies'].values)

    results = {
        'bodies': unique_body_num.tolist(),
        'threads': [],
        'mean': [],
        'lower': [],
        'upper': [],
    }

    for n_bodies in results['bodies']:
        query_1 = data[data['nbodies'] == n_bodies]
        threads_for_bodies = np.unique(query_1['threads'].values)
        results['threads'].append(threads_for_bodies)
        mean = []
        lower = []
        upper = []

        for n_threads in threads_for_bodies:
            subset = query_1[query_1['threads'] == n_threads]
            m, l, u = get_mean_ci(subset['elapsed'].values)
            mean.append(m)
            lower.append(l)
            upper.append(u)

        results['lower'].append(lower.copy())
        results['mean'].append(mean.copy())
        results['upper'].append(upper.copy())

    return results


if __name__ == '__main__':
    results = compute_results('./results_static.csv')

    fig_colors = [
        ('forestgreen', 'mediumseagreen'),  # 1024
        ('navy', 'lightskyblue'),  # 2048
        ('gold', 'blanchedalmond'),  # 4096
        ('firebrick', 'lightcoral'),  # 8192
    ]

    assert len(fig_colors) == len(results['bodies'])

    fig, ax = plot_complete(results, n_iter=1000, colors=fig_colors)
