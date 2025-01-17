import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms

import dgl

import sys, argparse, os, copy, itertools, glob, datetime
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score


def get_bag_feats(csv_file_df, args):
    if args.dataset == 'TCGA-lung-default':
        feats_csv_path = 'datasets/tcga-dataset/tcga_lung_data_feats/' + csv_file_df.iloc[0].split('/')[1] + '.csv'
        edges_csv_path = 'datasets/tcga-dataset/tcga_lung_data_edges_partial/edges_' + csv_file_df.iloc[0].split('/')[1] + '.csv'
    else:
        feats_csv_path = csv_file_df.iloc[0]
        splt = feats_csv_path.split('/')
        edges_csv_path = '/'.join(splt[:-2]) + '/edges_partial/edges_' + splt[-1]

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

    return label, graph, feats

# TODO?: MAKE WORK FOR GRAPH
def dropout_patches(feats, p):
    idx = np.random.choice(np.arange(feats.shape[0]), int(feats.shape[0]*(1-p)), replace=False)
    sampled_feats = np.take(feats, idx, axis=0)
    pad_idx = np.random.choice(np.arange(sampled_feats.shape[0]), int(feats.shape[0]*p), replace=False)
    pad_feats = np.take(sampled_feats, pad_idx, axis=0)
    sampled_feats = np.concatenate((sampled_feats, pad_feats), axis=0)
    return sampled_feats

def train(train_df, milnet, criterion, optimizer, args):
    # Set model to training mode
    milnet.train()

    if torch.cuda.is_available():
        Tensor = torch.cuda.FloatTensor
    else:
        Tensor = torch.FloatTensor
    
    total_loss = 0
    Tensor = torch.cuda.FloatTensor
    for i in range(len(train_df)):
        optimizer.zero_grad()
        label, graph, feats = get_bag_feats(train_df.iloc[i], args)  # Feats is a graph

        # TODO?: DROPOUT NOT IMPLEMENTED YET
        # feats = dropout_patches(feats, args.dropout_patch)

        # Variable --> torch.autograd
        bag_label = Variable(torch.FloatTensor([label]))
        # GPU: bag_label = Variable(torch.cuda.FloatTensor([label]))
        bag_graph = graph.to('cuda:0')
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
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()
        sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f' % (i, len(train_df), loss.item()))
    return total_loss / len(train_df)

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def multi_label_roc(labels, predictions, num_classes, pos_label=1):
    fprs = []
    tprs = []
    thresholds = []
    thresholds_optimal = []
    aucs = []
    if len(predictions.shape)==1:
        predictions = predictions[:, None]
    for c in range(0, num_classes):
        label = labels[:, c]
        prediction = predictions[:, c]
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        try:
            c_auc = roc_auc_score(label, prediction)
        except:
            print("Only one class present in y_true. ROC AUC score is not defined in that case.")
            c_auc=1
        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)
    return aucs, thresholds, thresholds_optimal

def test(test_df, milnet, criterion, args):
    milnet.eval()
    
    if torch.cuda.is_available():
        Tensor = torch.cuda.FloatTensor
    else:
        Tensor = torch.FloatTensor

    total_loss = 0
    test_labels = []
    test_predictions = []
    with torch.no_grad():
        for i in range(len(test_df)):
            label, graph, feats = get_bag_feats(test_df.iloc[i], args)  # Feats is a graph

            # TODO?: DROPOUT NOT IMPLEMENTED YET
            # feats = dropout_patches(feats, args.dropout_patch)

            # Variable --> torch.autograd
            bag_label = Variable(torch.FloatTensor([label]))
            # GPU: bag_label = Variable(torch.cuda.FloatTensor([label]))
            bag_graph = graph.to('cuda:0')
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
    auc_value, _, thresholds_optimal = multi_label_roc(test_labels, test_predictions, args.num_classes, pos_label=1)
    if args.num_classes==1:
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions>=thresholds_optimal[0]] = 1
        class_prediction_bag[test_predictions<thresholds_optimal[0]] = 0
        test_predictions = class_prediction_bag
        test_labels = np.squeeze(test_labels)
    else:        
        for i in range(args.num_classes):
            class_prediction_bag = copy.deepcopy(test_predictions[:, i])
            class_prediction_bag[test_predictions[:, i]>=thresholds_optimal[i]] = 1
            class_prediction_bag[test_predictions[:, i]<thresholds_optimal[i]] = 0
            test_predictions[:, i] = class_prediction_bag
    bag_score = 0
    for i in range(0, len(test_df)):
        bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score         
    avg_score = bag_score / len(test_df)
    
    return total_loss / len(test_df), avg_score, auc_value, thresholds_optimal

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
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--non_linearity', default=1, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--average', type=bool, default=True, help='Average the score of max-pooling and bag aggregating')

    # Graph conv arguments
    parser.add_argument('--model', default='graph_dsmil', type=str, help='MIL model [dsmil|graph_dsmil]')
    parser.add_argument('--gcn_layer_type', default='GraphConv', type=str, help='Type of GCN layer to use in model [GraphConv|GATConv|SAGEConv]')
    parser.add_argument('--n_gcn_layers', default=1, type=int, help='Number of GCN (or other graph type) layers to apply')
    parser.add_argument('--agg_type', default='dsmil', type=str, help='Aggregator type to use [dsmil|GlobalAttentionPooling]')

    # python3 TMP_train_gcn_agg.py --dataset=Camelyon16 --num_epochs=2 --model=graph_dsmil --gcn_layer_type=GraphConv --n_gcn_layers=1 --agg_type=dsmil

    args = parser.parse_args()
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)
    
    # CREATE MODEL
    if args.model == 'graph_dsmil':
        import graph_dsmil as mil

        # TODO: MAKE MATCH ARGS FOR "graph_dsmil.py"
        graph_module = mil.GraphModule(layer_type=args.gcn_layer_type, n_layers=args.n_gcn_layers).cuda()

        if args.agg_type == 'dsmil':
            i_classifier = mil.FCLayer(in_size=args.feats_size, out_size=args.num_classes).cuda()
            b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_classes, dropout_v=args.dropout_node, nonlinear=args.non_linearity).cuda()
            agg_module = mil.DSMILAgg(i_classifier, b_classifier).cuda()
        elif args.agg_type == 'GlobalAttentionPooling':
            # TODO: IMPLEMENT
            raise NotImplementedError('GlobalAttentionPooling aggregator is not yet implemented')
            agg_module = mil.GraphAttnAgg().cuda()
        else:
            raise ValueError('--agg_type must be "dsmil" or "GlobalAttentionPooling"')

        milnet = mil.GRAPH_MILNet(graph_module, agg_module).cuda()

    elif args.model == 'dsmil':
        import dsmil as mil

        i_classifier = mil.FCLayer(in_size=args.feats_size, out_size=args.num_classes).cuda()
        b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_classes, dropout_v=args.dropout_node, nonlinear=args.non_linearity).cuda()
        milnet = mil.MILNet(i_classifier, b_classifier).cuda()

        state_dict_weights = torch.load('init.pth')
        try:
            milnet.load_state_dict(state_dict_weights, strict=False)
        except:
            del state_dict_weights['b_classifier.v.1.weight']
            del state_dict_weights['b_classifier.v.1.bias']
            milnet.load_state_dict(state_dict_weights, strict=False)

    # LOSS AND TRAINING
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)
    
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
    save_path = os.path.join('weights', datetime.date.today().strftime("%m%d%Y"))
    os.makedirs(save_path, exist_ok=True)
    run = len(glob.glob(os.path.join(save_path, '*.pth')))
    for epoch in range(1, args.num_epochs):
        train_path = shuffle(train_path).reset_index(drop=True)
        test_path = shuffle(test_path).reset_index(drop=True)

        # Train for 1 epoch
        train_loss_bag = train(train_path, milnet, criterion, optimizer, args) # iterate all bags

        # Evaluate
        test_loss_bag, avg_score, aucs, thresholds_optimal = test(test_path, milnet, criterion, args)
        if args.dataset.startswith('TCGA-lung'):
            print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, auc_LUAD: %.4f, auc_LUSC: %.4f' % 
                  (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score, aucs[0], aucs[1]))
        else:
            print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, AUC: ' % 
                  (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs))) 
        
        scheduler.step()
        
        # Save results for best model version/epoch
        current_score = (sum(aucs) + avg_score)/2
        if current_score >= best_score:
            best_score = current_score
            save_name = os.path.join(save_path, str(run+1)+'.pth')
            torch.save(milnet.state_dict(), save_name)
            if args.dataset.startswith('TCGA-lung'):
                print('Best model saved at: ' + save_name + ' Best thresholds: LUAD %.4f, LUSC %.4f' % (thresholds_optimal[0], thresholds_optimal[1]))
            else:
                print('Best model saved at: ' + save_name)
                print('Best thresholds ===>>> '+ '|'.join('class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal)))
            

if __name__ == '__main__':
    main()