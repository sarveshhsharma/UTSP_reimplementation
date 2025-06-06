import torch
import torch.nn.functional as F
from torch import tensor
import torch.nn
from diff_moduleS4p import scattering_diffusionS4,GCN_diffusion
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn as nn
from torch_geometric.nn import GATConv

class GATModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, heads=1):
        super(GATModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList()

        for i in range(n_layers):
            in_channels = hidden_dim if i == 0 else hidden_dim * heads
            self.convs.append(GATConv(in_channels, hidden_dim, heads=heads, concat=True))

        self.mlp1 = nn.Linear(hidden_dim * heads, output_dim) #out_proj
        self.m = nn.Softmax(dim=1) #softmax

    def forward(self, X, adj):
        # Use in_proj to map input coordinates
        X = self.in_proj(X)

        # Convert full adj matrix to edge_index (sparse format for GATConv)
        edge_index = (adj > 0).nonzero(as_tuple=False).t().contiguous()  # shape: [2, num_edges]

        for gat_layer in self.convs:
            X = F.elu(gat_layer(X, edge_index))

        output = self.mlp1(X)
        output = self.m(output)

        return output
