import argparse, os, itertools
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

    print(feats_csv_path)
    print(edges_csv_path)
    print()

    # Get bag in dataframe
    df = pd.read_csv(feats_csv_path)

    feats = df.to_numpy()  # [[patch embedding], [patch embedding], ...]
    src = deque()
    dst = deque()
    weights = deque()
    ids = list(df.index)
    for id_a, id_b in itertools.combinations(ids, 2):
        dist = 1 - cosine(feats[id_a], feats[id_b])
        
        # Append edge twice because undirected
        src.append(id_a)
        dst.append(id_b)
        weights.append(dist)

        src.append(id_b)
        dst.append(id_a)
        weights.append(dist)

    # TODO: CHECK WEIGHT IS CORRECT FOR GIVEN SRC/DST PAIR
    edge_df = pd.DataFrame({'Src': src, 'Dst': dst, 'Weight': weights})

    df.to_csv(feats_csv_path, index=True)
    edge_df.to_csv(edges_csv_path, index=True)


def main():
    # ARGUMENTS
    parser = argparse.ArgumentParser(description='Train DSMIL on 20x patch features learned by SimCLR')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--feats_size', default=512, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--lr', default=0.0002, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=200, type=int, help='Number of total training epochs [40|200]')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--dataset', default='TCGA-lung-default', type=str, help='Dataset folder name')
    parser.add_argument('--split', default=0.2, type=float, help='Training/Validation split [0.2]')
    parser.add_argument('--model', default='dsmil', type=str, help='MIL model [dsmil]')
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--non_linearity', default=1, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--average', type=bool, default=True, help='Average the score of max-pooling and bag aggregating')
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
    for i in tqdm(range(len(paths))):
        get_ids_and_edges(paths.iloc[i], args)

if __name__ == '__main__':
    main()
    