import argparse, os
from collections import deque
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cosine
from sklearn.utils import shuffle
from sklearn.neighbors import NearestNeighbors


def get_ids_and_edges(csv_file_df, args):
    n_neigh_list = [2, 4, 8, 16, 32]

    if args.dataset == 'TCGA-lung-default':
        feats_csv_path = 'datasets/tcga-dataset/tcga_lung_data_feats/' + csv_file_df.iloc[0].split('/')[1] + '.csv'
        edges_csv_paths = [(f'datasets/tcga-dataset/tcga_lung_data_edges_{n}/edges_' + csv_file_df.iloc[0].split('/')[1] + '.csv') for n in n_neigh_list]
    else:
        feats_csv_path = csv_file_df.iloc[0]
        splt = feats_csv_path.split('/')
        edges_csv_paths = [('/'.join(splt[:-2]) + f'/edges_{n}/edges_' + splt[-1]) for n in n_neigh_list]

    for dir in edges_csv_paths:
        dir_tmp = '/'.join(dir.split('/')[:-1])
        if not os.path.exists(dir_tmp):
            os.mkdir(dir_tmp)

    print()
    print(feats_csv_path)
    print(edges_csv_paths)

    # Get bag in dataframe
    df = pd.read_csv(feats_csv_path)
    if len(df.columns) > 512:
        df = df.set_index('Unnamed: 0', drop=True)

    # Save so index is available
    df.to_csv(feats_csv_path, index=True)

    feats = df.to_numpy()  # [[patch embedding], [patch embedding], ...]
    ids = list(df.index)
    print(len(ids))

    # Doing 2, 4, 8, 16, 32 neighbors
    print('Fitting nearest neighbors')
    # Make sure n_neigh is not larger than num samples
    calc_neigh_n = min(len(ids), 32)
    nbrs = NearestNeighbors(n_neighbors=calc_neigh_n, algorithm='ball_tree').fit(feats)

    src = {n:deque() for n in n_neigh_list}
    dst = {n:deque() for n in n_neigh_list}
    weights = {n:deque() for n in n_neigh_list}
    for id_a in tqdm(ids):
        distances, indices = nbrs.kneighbors([feats[id_a]])
        indices = indices[0]
        distances = distances[0]

        zipped_i_d = [(i, d) for i, d in zip(indices, distances)]
        # Sort by distance
        zipped_i_d = sorted(zipped_i_d, key=lambda x: x[1])
        # Convert to cosine distance
        zipped_i_d = [(i, 1 - cosine(feats[id_a], feats[i])) for i, _ in zipped_i_d]

        for n in n_neigh_list:
            neigh_list = zipped_i_d[:n]
            for id_b, dist in neigh_list:
                src[n].append(id_a)
                dst[n].append(id_b)
                weights[n].append(dist)

    for nn, f_path in zip(n_neigh_list, edges_csv_paths):
        edge_df = pd.DataFrame({'Src': src[nn], 'Dst': dst[nn], 'Weight': weights[nn]})

        edge_df.to_csv(f_path, index=True)


def main():
    # ARGUMENTS
    parser = argparse.ArgumentParser(description='Train DSMIL on 20x patch features learned by SimCLR')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--feats_size', default=512, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--dataset', default='TCGA-lung-default', type=str, help='Dataset folder name')
    parser.add_argument('--start', default=0, type=int, help='use to only run for partial dataset')
    parser.add_argument('--end', default=1048, type=int, help='use to only run for partial dataset')
    args = parser.parse_args()
    
    # DATASET
    if args.dataset == 'TCGA-lung-default':
        bags_csv = 'datasets/tcga-dataset/TCGA.csv'
    else:
        bags_csv = os.path.join('datasets', args.dataset, args.dataset+'.csv')
        
    bags_path = pd.read_csv(bags_csv)
    bags_path = bags_path.iloc[args.start:args.end, :]

    paths = shuffle(bags_path).reset_index(drop=True)
    for i in range(len(paths)):
        get_ids_and_edges(paths.iloc[i], args)

if __name__ == '__main__':
    main()
    