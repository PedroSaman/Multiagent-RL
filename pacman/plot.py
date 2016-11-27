#!/usr/bin/python
# -*- coding: utf-8 -*-

u"""Process results for simulations and genarates plots.

Attributes:
    T: Dict of behaviors {'FleeBehavior': u'Fuga',
        'PursueBehavior': u'Perseguição', 'SeekBehavior': u'Busca',}
    COLOR_TABLE:  Dict of colors, { 'r': '#c0392b', 'g': '#16a085',
        'b': '#2980b9', 'y': '#f39c12', 'p': '#8e44ad', 'w': '#ecf0f1',
        'k': '#2c3e50',}
    COLOR_LIST: List of color values, ['r', 'g', 'b', 'y', 'p', 'w', 'k'].
"""

from __future__ import division
import argparse
import pickle
import matplotlib.pylab as plt
import numpy as np

__author__ = "Matheus Portela and Guilherme N. Ramos"
__credits__ = ["Matheus Portela", "Guilherme N. Ramos", "Renato Nobre",
               "Pedro Saman"]
__maintainer__ = "Guilherme N. Ramos"
__email__ = "gnramos@unb.br"

T = {
    'FleeBehavior': u'Fuga',
    'PursueBehavior': u'Perseguição',
    'SeekBehavior': u'Busca',
}

COLOR_TABLE = {
    'r': '#c0392b',
    'g': '#16a085',
    'b': '#2980b9',
    'y': '#f39c12',
    'p': '#8e44ad',
    'w': '#ecf0f1',
    'k': '#2c3e50',
}

COLOR_LIST = ['r', 'g', 'b', 'y', 'p', 'w', 'k']


def load_results(filename):
    """Load results.

    Load results from a output file

    Args:
        filename: The name of the file.
    Returns:
        results: The results in the file
    """
    with open(filename) as f:
        results = pickle.loads(f.read())
    return results


def calculate_regression_coefficients(data, degree=4):
    """Calculate the regression coefficients.

    Args:
        data: The data to perform the calculation.
        degree: Degree of the fitting polynomial
    Returns:
        Regression coefficients.
    """
    return np.polyfit(range(len(data)), data, degree)


def calculate_regression_y(x, coeff):
    """Calculate the regression.

    Args:
        coeff: The coefficients for each regression.
    Returns:
        Sum of array elements over a given axis.
    """
    return np.sum([x**(len(coeff) - i - 1) *
                   coeff[i] for i in range(len(coeff))])


def plot_scores(learn_scores, test_scores):
    """Create a plot for the scores.

    Args:
        learn_scores: The score from the learn runs.
        test_scores: The score from the test runs.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlabel(u'Número de jogos')
    plt.ylabel(u'Pontuação final')
    plt.title(u'Pontuação final ao longo dos jogos')
    plt.xlim([0, 115])
    data = learn_scores + test_scores
    coeff = calculate_regression_coefficients(data, degree=1)
    regression = [calculate_regression_y(x, coeff) for x in range(len(data))]

    print 'Regression coefficients:', coeff

    ax.scatter(range(len(learn_scores)), learn_scores,
               c=COLOR_TABLE['r'], marker='o')
    ax.scatter(range(len(learn_scores), len(learn_scores) + len(test_scores)),
               test_scores, c=COLOR_TABLE['g'], marker='D')
    ax.plot(regression, c=COLOR_TABLE['b'], linewidth=2.0)


def plot_game_duration(behavior_count):
    """Create a plot for the game duration.

    Args:
        behavior_count: The count of a behavior.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlabel(u'Número de jogos')
    plt.ylabel(u'Número de instantes de jogo')
    plt.title(u'Duração dos jogos')
    plt.xlim([0, 115])

    data = np.sum(np.array(
        [np.array(b) for b in behavior_count.values()[0].values()]), axis=0)

    coeff = calculate_regression_coefficients(data, degree=1)
    regression = [calculate_regression_y(x, coeff) for x in range(len(data))]
    ax.scatter(range(len(data)), data, c=COLOR_TABLE['b'])
    ax.plot(regression, c=COLOR_TABLE['r'])

    ax.legend()


def plot_behavior_count(agent_id, behavior_count):
    """Create a plot for the behavior count.

    Args:
        agent_id: The identifier of the agent.
        behavior_count: The count of the agent behaviors.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlabel(u'Número de jogos')
    plt.ylabel(u'Probabilidades de selecionar comportamento')
    plt.title(u'Probabilidades do agente %d selecionar comportamento'
              % agent_id)

    plt.xlim([0, 115])
    plt.ylim([-0.1, 1.1])

    data = np.array([b for b in behavior_count.values()])
    prob = data/np.sum(data, axis=0)

    for i, behavior in enumerate(behavior_count):
        coeff = calculate_regression_coefficients(prob[i], degree=4)
        regression = [calculate_regression_y(x, coeff)
                      for x in range(len(prob[i]))]
        ax.plot(regression, label=T[behavior],
                c=COLOR_TABLE[COLOR_LIST[i]], linewidth=2.0)
        ax.scatter(range(len(prob[i])), prob[i], c=COLOR_TABLE[COLOR_LIST[i]], alpha=1)

    ax.legend()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Pacman simulations.')
    parser.add_argument('-i', '--input', dest='input_filename', type=str,
                        default='results.txt', help='results input file')

    args = parser.parse_args()

    results_input_filename = args.input_filename

    results = load_results(results_input_filename)

    plot_scores(results['learn_scores'], results['test_scores'])
    plot_game_duration(results['behavior_count'])
    for agent_id, behavior_count in results['behavior_count'].items():
        plot_behavior_count(agent_id, behavior_count)
    plt.show()
