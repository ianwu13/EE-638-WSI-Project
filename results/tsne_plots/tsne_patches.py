import time

import numpy as np
import pandas as pd

from sklearn.manifold import TSNE


norm_patch_pth = '../../datasets/Camelyon16/0-normal/normal_085.csv'
norm_tsne_output = './norm_tsne_vals.txt'
tumo_patch_pth = '../../datasets/Camelyon16/1-tumor/tumor_005.csv'
tumo_tsne_output = './tumo_tsne_vals.txt'


def tsne_plot(data, out_file):
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data)

    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    
    tsne_2d_one = tsne_results[:,0]
    tsne_2d_two = tsne_results[:,1]

    for v in tsne_2d_one:
        out_file.write(f'{v}, ')
    out_file.write('\n')
    for v in tsne_2d_two:
        out_file.write(f'{v}, ')


for num in [71, 72, 73, 77, 78, 65]:
    tumo_patch_pth = f'../../datasets/Camelyon16/1-tumor/tumor_0{num}.csv'
    tumo_tsne_output = f'./{num}.txt'
    dat = pd.read_csv(tumo_patch_pth, index_col=0).values
    f = open(tumo_tsne_output, 'w')
    tsne_plot(dat, f)
    f.close()
    print('RAN TSNE FOR NORM WSI')

exit()

# OLD
tumo_patch_pth = '../../datasets/Camelyon16/1-tumor/tumor_038.csv'
tumo_tsne_output = './38.txt'
dat = pd.read_csv(tumo_patch_pth, index_col=0).values
f = open(tumo_tsne_output, 'w')
tsne_plot(dat, f)
f.close()
print('RAN TSNE FOR NORM WSI')

tumo_patch_pth = '../../datasets/Camelyon16/1-tumor/tumor_039.csv'
tumo_tsne_output = './39.txt'
dat = pd.read_csv(tumo_patch_pth, index_col=0).values
f = open(tumo_tsne_output, 'w')
tsne_plot(dat, f)
f.close()
print('RAN TSNE FOR NORM WSI')

tumo_patch_pth = '../../datasets/Camelyon16/1-tumor/tumor_041.csv'
tumo_tsne_output = './41.txt'
dat = pd.read_csv(tumo_patch_pth, index_col=0).values
f = open(tumo_tsne_output, 'w')
tsne_plot(dat, f)
f.close()
print('RAN TSNE FOR NORM WSI')

tumo_patch_pth = '../../datasets/Camelyon16/1-tumor/tumor_042.csv'
tumo_tsne_output = './42.txt'
dat = pd.read_csv(tumo_patch_pth, index_col=0).values
f = open(tumo_tsne_output, 'w')
tsne_plot(dat, f)
f.close()
print('RAN TSNE FOR NORM WSI')

tumo_patch_pth = '../../datasets/Camelyon16/1-tumor/tumor_043.csv'
tumo_tsne_output = './43.txt'
dat = pd.read_csv(tumo_patch_pth, index_col=0).values
f = open(tumo_tsne_output, 'w')
tsne_plot(dat, f)
f.close()
print('RAN TSNE FOR NORM WSI')

tumo_patch_pth = '../../datasets/Camelyon16/1-tumor/tumor_049.csv'
tumo_tsne_output = './49.txt'
dat = pd.read_csv(tumo_patch_pth, index_col=0).values
f = open(tumo_tsne_output, 'w')
tsne_plot(dat, f)
f.close()
print('RAN TSNE FOR NORM WSI')

dat = pd.read_csv(norm_patch_pth, index_col=0).values
f = open(norm_tsne_output, 'w')
tsne_plot(dat, f)
f.close()
print('RAN TSNE FOR NORM WSI')

dat = pd.read_csv(tumo_patch_pth, index_col=0).values
f = open(tumo_tsne_output, 'w')
tsne_plot(dat, f)
f.close()
print('RAN TSNE FOR TUMO WSI')
