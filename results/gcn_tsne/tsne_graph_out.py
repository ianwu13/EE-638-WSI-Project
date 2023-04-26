import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms

import dgl

import sys, argparse, os, copy, itertools, glob, datetime, random, warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score

import tsne_graph_dsmil as mil


def get_bag_feats_graph(csv_file_df, edges_per_node, args):
    if args.dataset == 'TCGA-lung-default':
        feats_csv_path = 'datasets/tcga-dataset/tcga_lung_data_feats/' + csv_file_df.iloc[0].split('/')[1] + '.csv'
        edges_csv_path = f'datasets/tcga-dataset/tcga_lung_data_edges_{edges_per_node}/edges_' + csv_file_df.iloc[0].split('/')[1] + '.csv'
    else:
        feats_csv_path = csv_file_df.iloc[0]
        splt = feats_csv_path.split('/')
        edges_csv_path = '/'.join(splt[:-2]) + f'/edges_{edges_per_node}/edges_' + splt[-1]

    # Get label for sample
    label = np.zeros(args.num_classes)
    if args.num_classes==1:
        label[0] = csv_file_df.iloc[1]
    else:
        if int(csv_file_df.iloc[1])<=(len(label)-1):
            label[int(csv_file_df.iloc[1])] = 1

    # Get features in dataframe
    node_df = pd.read_csv(feats_csv_path, index_col=0)
    feats = node_df.to_numpy()
    
    edge_df = pd.read_csv(edges_csv_path, index_col=0)

    # Create a DGL graph
    src = edge_df['Src'].to_numpy()
    dst = edge_df['Dst'].to_numpy()
    graph = dgl.graph((src, dst))

    # Add edge weights
    edge_weight = torch.tensor(edge_df['Weight'].to_numpy())
    graph.edata['weight'] = edge_weight

    # Add node features
    for col in node_df.columns:
        data = torch.tensor(node_df[col].to_numpy())
        graph.ndata[col] = data

    # Add self loops
    graph = dgl.add_self_loop(graph)

    return label, graph, feats

def test(test_df, milnet, edges_per_node, criterion, args):
    milnet.eval()
    '''
    if torch.cuda.is_available():
        Tensor = torch.cuda.FloatTensor
    else:
        Tensor = torch.FloatTensor
    '''
    Tensor = torch.FloatTensor

    total_loss = 0
    test_labels = []
    test_predictions = []
    with torch.no_grad():
        for i in range(len(test_df)):
            label, graph, feats = get_bag_feats_graph(test_df.iloc[i], edges_per_node, args)  # Feats is a graph

            # TODO?: DROPOUT NOT IMPLEMENTED YET
            # feats = dropout_patches(feats, args.dropout_patch)

            # Variable --> torch.autograd
            bag_label = Variable(torch.FloatTensor([label]))
            # GPU: bag_label = Variable(torch.cuda.FloatTensor([label]))
            bag_graph = graph  # .to('cuda:0')
            # GPU: bag_graph = Variable(feats.to('cuda:0'))
            bag_feats = Variable(Tensor([feats]))
            bag_feats = bag_feats.view(-1, args.feats_size)
            
            # TODO: NOT SURE WHAT THIS DOES: bag_graph = bag_graph.view(-1, args.feats_size)

            # TODO: UPDATE "graph_dsmil.py" to include graph stuff
            ins_prediction, bag_prediction, _, _ = milnet(bag_graph, bag_feats)

            max_prediction, _ = torch.max(ins_prediction, 0)  
            bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
            loss = 0.5*bag_loss + 0.5*max_loss
            total_loss = total_loss + loss.item()
            sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (i, len(test_df), loss.item()))
            test_labels.extend([label])
            if args.average:
                test_predictions.extend([(0.5*torch.sigmoid(max_prediction)+0.5*torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])
            else: test_predictions.extend([(0.0*torch.sigmoid(max_prediction)+1.0*torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])
    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)

def main():
    # Surpress warnings
    warnings.filterwarnings('ignore')

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
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--non_linearity', default=1, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--average', type=bool, default=True, help='Average the score of max-pooling and bag aggregating')
    args = parser.parse_args()
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)
    
    # CREATE MODEL
    graph_module = mil.GraphModule(layer_type='GraphConv', n_layers=2)  #  .cuda()
    i_classifier = mil.FCLayer(in_size=args.feats_size, out_size=args.num_classes)  #  .cuda()
    b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_classes, dropout_v=args.dropout_node, nonlinear=args.non_linearity)  #  .cuda()
    agg_module = mil.DSMILAgg(i_classifier, b_classifier)  #  .cuda()
    milnet = mil.GRAPH_MILNet(graph_module, agg_module)  #  .cuda()

    # DATASET
    if args.dataset == 'TCGA-lung-default':
        bags_csv = 'datasets/tcga-dataset/TCGA.csv'
    else:
        bags_csv = os.path.join('datasets', args.dataset, args.dataset+'.csv')
    bags_path = pd.read_csv(bags_csv)
    train_path = bags_path.iloc[0:int(len(bags_path)*(1-args.split)), :]
    print(f'N Training Samples: {len(train_path)}')
    test_path = bags_path.iloc[int(len(bags_path)*(1-args.split)):, :]
    print(f'N Testing Samples: {len(test_path)}')

    # TRAIN MODEL
    best_score = 0
    for epoch in tqdm(range(1, args.num_epochs)):
        test_path = shuffle(test_path).reset_index(drop=True)

        # Evaluate
        test(test_path, milnet, 8, criterion, args)
        
    print('COMPLETED')       

if __name__ == '__main__':
    main()