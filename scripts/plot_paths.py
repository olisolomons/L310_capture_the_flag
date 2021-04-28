from matplotlib import pyplot as plt
import json
import os
import numpy as np


def load(team_size, number):
    path_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'path_log/%d_%d.json' % (team_size, number)
    )
    with open(path_file_path, 'r') as path_file:
        return np.array(json.load(path_file))


def path_length(path):
    return (np.linalg.norm(path[1:] - path[:-1], axis=1)).sum()


def plot_length(max_team_size):
    longest_path_length = []
    total_path_length = []
    for team_size in range(1, max_team_size + 1):
        paths = [path_length(load(team_size, i)) for i in range(team_size)]
        longest_path_length.append(max(paths))
        total_path_length.append(sum(paths))

    plt.plot(np.arange(max_team_size) + 1, longest_path_length, label='longest path length')
    plt.plot(np.arange(max_team_size) + 1, total_path_length, label='total path length')
    plt.legend(loc='lower left')
    plt.xlabel('number of robots')
    plt.ylabel('distance (m)')
    plt.show()


if __name__ == '__main__':
    plot_length(9)
