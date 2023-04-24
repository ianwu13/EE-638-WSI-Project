import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


norm_tsne_output = './norm_tsne_vals.txt'
tumo_tsne_output = './tumo_tsne_vals.txt'


def plot_tsne(dat_file_path, out_path):
    f = open(dat_file_path, 'r')
    dim_1 = [float(x) for x in f.readline().rstrip(', ').split(', ')]
    dim_2 = [float(x) for x in f.readline().rstrip(', ').split(', ')]
    f.close

    plot = sns.scatterplot(x=dim_1, y=dim_2)
    fig = plot.get_figure()
    fig.savefig(out_path)
    plt.clf()


plot_tsne(norm_tsne_output, 'norm_tsne.png')
plot_tsne(tumo_tsne_output, 'tumo_tsne.png')
