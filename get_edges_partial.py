import argparse, os, random
from collections import deque
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cosine
from sklearn.utils import shuffle


def get_ids_and_edges(csv_file_df, args):
    if args.dataset == 'TCGA-lung-default':
        feats_csv_path = 'datasets/tcga-dataset/tcga_lung_data_feats/' + csv_file_df.iloc[0].split('/')[1] + '.csv'
        edges_csv_path = 'datasets/tcga-dataset/tcga_lung_data_edges/edges_' + csv_file_df.iloc[0].split('/')[1] + '.csv'
    else:
        feats_csv_path = csv_file_df.iloc[0]
        splt = feats_csv_path.split('/')
        edges_csv_path = '/'.join(splt[:-2]) + '/edges/edges_' + splt[-1]

    print()
    print(feats_csv_path)
    print(edges_csv_path)

    # Get bag in dataframe
    df = pd.read_csv(feats_csv_path)
    if len(df.columns) > 512:
        df = df.set_index('Unnamed: 0', drop=True)

    feats = df.to_numpy()  # [[patch embedding], [patch embedding], ...]
    src = deque()
    dst = deque()
    weights = deque()
    ids = list(df.index)
    print(len(ids))
    for id_a in tqdm(ids):
        for id_b in set(random.choices(ids, k=100)):
            dist = 1 - cosine(feats[id_a], feats[id_b])
            
            # Append edge twice because undirected
            src.append(id_a)
            dst.append(id_b)
            weights.append(dist)

    edge_df = pd.DataFrame({'Src': src, 'Dst': dst, 'Weight': weights})

    df.to_csv(feats_csv_path, index=True)
    edge_df.to_csv(edges_csv_path, index=True)


def main():
    # ARGUMENTS
    parser = argparse.ArgumentParser(description='Train DSMIL on 20x patch features learned by SimCLR')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--feats_size', default=512, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--dataset', default='TCGA-lung-default', type=str, help='Dataset folder name')
    args = parser.parse_args()
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)
    
    # DATASET
    if args.dataset == 'TCGA-lung-default':
        bags_csv = 'datasets/tcga-dataset/TCGA.csv'
    else:
        bags_csv = os.path.join('datasets', args.dataset, args.dataset+'.csv')
        
    bags_path = pd.read_csv(bags_csv)
    bags_path = bags_path.iloc[0:, :]

    paths = shuffle(bags_path).reset_index(drop=True)
    for i in range(len(paths)):
        get_ids_and_edges(paths.iloc[i], args)

if __name__ == '__main__':
    main()
    