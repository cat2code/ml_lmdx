import torch
import torch.nn as nn
from torch_geometric.nn import GraphConv, global_mean_pool


class SimpleGNN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 32, out_dim: int = 2):
        super().__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return self.head(x)