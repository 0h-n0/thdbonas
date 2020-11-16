import torch
import torch.nn as nn
import torchex.nn as exnn

from .gcn_conv import LazyGCNConv
from torch_geometric.nn import global_mean_pool


class ChemblGCNConv(nn.Module):
    def __init__(self, out_channels):
        super(ChemblGCNConv, self).__init__()
        self.out_channels = out_channels
        self.gcn = LazyGCNConv(out_channels)

    def forward(self, data):
        adj, edge_indices = data[0], data[1].edge_index
        x = self.gcn(adj, edge_index)
        return (x, data[1])
    

class ChemblGlobalMeanPool(nn.Module):
    def __init__(self):
        super(ChemblGlobalMeanPool, self).__init__()

    def __init__(self, data):
        x = data[0]
        batch = data[1].batch
        x = global_mean_pool(x, batch)
        return (x, data[1])


class ChemblLinear(nn.Module):
    def __init__(self, out_channels):
        super(ChemblLinear, self).__init__()
        self.out_channels = out_channels        
        self.l = exnn.Linear(out_channels)
        self.relu = nn.ReLU()

    def forward(self, data):
        x = data[0]
        x = self.relu(x)
        return (x, data[1])


class ChemblFlattenLinear(nn.Module):
    def __init__(self, out_channels):
        super(ChemblFlattenLinear, self).__init__()
        self.out_channels = out_channels
        self.linear = exnn.Linear(self.out_channels)
        self.relu = nn.ReLU()

    def forward(self, data):
        x = data[0]        
        if len(x.shape) == 4:
            B, _, _, _ = x.shape
            x = x.reshape(B, -1)
        return (self.relu(self.linear(x)), data[1])

    
class ChemblConcatFlatten(nn.Module):
    def __init__(self, out_channels):
        super(ChemblConcatFlatten, self).__init__()
        self.out_channels = out_channels
        self.linear = exnn.Linear(self.out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, two_data):
        x1 = two_data[0][0]
        x2 = two_data[1][0]        
        x1 = torch.cat([x1, x2], dim=1)
        if len(x1.shape) == 4:
            B, _, _, _ = x1.shape
            x1 = x1.reshape(B, -1)
        return (self.relu(self.linear(x1)), tow_data[0][1])

class ChemblConcat(nn.Module):
    def __init__(self):
        super(ChemblConcat, self).__init__()
        
    def forward(self, two_data):
        x1 = two_data[0][0]
        x2 = two_data[1][0]        
        x = torch.cat([x1, x2], dim=1)
        return (x, two_data[0][1])
    
if __name__ == '__main__':
    from kgcn_torch.datasets.examples.multimodal_chembl import MultiModalChemblDataset    
    from torch_geometric.data import DataLoader
    train_loader = DataLoader(MultiModalChemblDataset(), batch_size=16, shuffle=True)
    for batch_idx, data in enumerate(train_loader):
        x = (data.x.float(), data)
        x2 = (data.x.float(), data)
        print(data.x)
        print(data.edge_index)
        print(data.y)
        o = net(data)
        print(o)
        a
        
