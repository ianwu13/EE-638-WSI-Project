import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from dgl.nn.pytorch.glob import GlobalAttentionPooling
import dgl.nn.pytorch.conv as dgl_conv

class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))

    def forward(self, feats):
        x = self.fc(feats)
        return feats, x

class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()
        
        self.feature_extractor = feature_extractor      
        self.fc = nn.Linear(feature_size, output_class)
        
    def forward(self, x):
        device = x.device
        feats = self.feature_extractor(x) # N x K
        c = self.fc(feats.view(feats.shape[0], -1)) # N x C
        return feats.view(feats.shape[0], -1), c

class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=True, passing_v=False): # K, L, N
        super(BClassifier, self).__init__()
        if nonlinear:
            self.q = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh())
        else:
            self.q = nn.Linear(input_size, 128)
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.Linear(input_size, input_size),
                nn.ReLU()
            )
        else:
            self.v = nn.Identity()
        
        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)  
        
    def forward(self, feats, c): # N x K, N x C
        device = feats.device
        V = self.v(feats) # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1) # N x Q, unsorted
        
        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0, descending=True) # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :]) # select critical instances, m_feats in shape C x K 
        q_max = self.q(m_feats) # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0, 1)) # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C, 
        B = torch.mm(A.transpose(0, 1), V) # compute bag representation, B in shape C x V
                
        B = B.view(1, B.shape[0], B.shape[1]) # 1 x C x V
        C = self.fcc(B) # 1 x C x 1
        C = C.view(1, -1)
        return C, A, B 
    
# TODO: HERE DOWN

class GraphModule(nn.Module):
    def __init__(self, layer_type, n_layers, n_feats=512, gatconv_n_heads=1):
        super(GraphModule, self).__init__()
        # layer_type: GraphConv|GATConv|SAGEConv
        self.layer_type = layer_type

        if layer_type == 'GraphConv':
            self.graph_layers = nn.ModuleList([dgl_conv.GraphConv(n_feats, n_feats) for _ in range(n_layers)])
        elif layer_type == 'GATConv':
            self.graph_layers = nn.ModuleList([dgl_conv.GATConv(n_feats, n_feats, gatconv_n_heads) for _ in range(n_layers)])
        elif layer_type == 'SAGEConv':
            raise NotImplementedError('GraphSAGE layer has not yet been implemented')
            self.graph_layers = nn.ModuleList([dgl_conv.SAGEConv(n_feats, n_feats) for _ in range(n_layers)])
        else:
            raise ValueError('"layer_type" must be in [GraphConv|GATConv|*SAGEConv]')

    def forward(self, g, feats):
        for lay in self.graph_layers:
            feats = lay(g, feats)

            if self.layer_type == 'GATConv':
                feats = torch.flatten(feats, start_dim=1, end_dim=2)

        return feats

class DSMILAgg(nn.Module):
    def __init__(self, i_classifier, b_classifier):
        super(DSMILAgg, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier
        
    def forward(self, x):
        feats, classes = self.i_classifier(x)
        prediction_bag, A, B = self.b_classifier(feats, classes)
        
        return classes, prediction_bag, A, B

class GraphAttnAgg(nn.Module):
    def __init__(self, input_size, output_class):
        super(GraphAttnAgg, self).__init__()
        self.fc = nn.Sequential(nn.Linear(input_size, output_class))  # Project each patch to a single value

        self.gate_nn = nn.Linear(input_size, 1)  # the gate layer that maps node feature to scalar
        self.gap = GlobalAttentionPooling(self.gate_nn)
        self.pooler = nn.Linear(input_size, output_class)

    def forward(self, graph, feats):
        classes = self.fc(feats)

        feats = self.gap(graph, feats)  # 512 (or num_feats) long

        pred = self.pooler(feats)
        
        return classes, pred, None, None  # Last 2 are unused

class GRAPH_MILNet(nn.Module):
    def __init__(self, graph_module, agg_module):
        super(GRAPH_MILNet, self).__init__()
        self.graph_module = graph_module

        if isinstance(agg_module, GraphAttnAgg):
            self.agg_type = 'graph'
        else:
            self.agg_type = 'bag'
        self.agg_module = agg_module
        
    def forward(self, g, feats):
        feats = self.graph_module(g, feats)

        if self.agg_type == 'graph':
            classes, prediction_bag, A, B = self.agg_module(g, feats)
        else:
            classes, prediction_bag, A, B = self.agg_module(feats)
        
        return classes, prediction_bag, A, B
        