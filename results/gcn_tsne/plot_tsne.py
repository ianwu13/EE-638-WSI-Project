import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


def plot_tsne(dat_file_path, out_path):
    f = open(dat_file_path, 'r')
    dim_1 = [float(x) for x in f.readline().rstrip(', \n').split(', ')]
    dim_2 = [float(x) for x in f.readline().rstrip(', ').split(', ')]
    f.close

    plot = sns.scatterplot(x=dim_1, y=dim_2)
    fig = plot.get_figure()
    fig.savefig(out_path)
    plt.clf()


plot_tsne('norm/65.txt', 'norm/65.png')
plot_tsne('norm/71.txt', 'norm/71.png')
plot_tsne('norm/72.txt', 'norm/72.png')
plot_tsne('norm/73.txt', 'norm/73.png')
plot_tsne('norm/77.txt', 'norm/77.png')
plot_tsne('norm/78.txt', 'norm/78.png')
