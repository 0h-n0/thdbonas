import torch
import torch.nn as nn
import torch.nn.functional as F
import torchex.nn as exnn

from .gcn_conv import LazyGCNConv
from torch_geometric.nn import global_mean_pool


class Tox21Value:
    def __init__(self, x, data):
        self._x = x
        self._data = data
        
    def size(self):
        return self._x.size()

    @property
    def x(self):
        pass

    @x.setter
    def x(self, x):
        self._x = x
        
    @x.getter
    def x(self):
        return self._x

    @property
    def data(self):
        pass

    @x.setter
    def data(self, data):
        self._data = data

    @x.getter
    def data(self):
        return self._data

class GlobalAvgPool1d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool1d, self).__init__()

    def forward(self, x):
        assert len(x.size()) == 3, x.size()
        B, C, L = x.size()
        x = F.avg_pool1d(x, L)
        return x
        

class Tox21Embedding(nn.Module):
    def __init__(self, seq_len=21, out_channels=64):
        super(Tox21Embedding, self).__init__()
        self.embd = nn.Embedding(seq_len, out_channels)
        self.pool = GlobalAvgPool1d()
        self.flatten = exnn.Flatten()

    def forward(self, value):
        x = value.x
        x = self.embd(x)
        x = self.pool(x)
        value.x = self.flatten(x)
        return value
    
        
class Tox21GCNConv(nn.Module):
    def __init__(self, out_channels):
        super(Tox21GCNConv, self).__init__()
        self.out_channels = out_channels
        self.gcn = LazyGCNConv(out_channels)
        self.relu = nn.ReLU()

    def forward(self, value):
        adj, edge_index = value.x.float(), value.data.edge_index
        x = self.gcn(adj, edge_index)
        value.x = self.relu(x)
        return value
    

class Tox21GlobalMeanPool(nn.Module):
    def __init__(self):
        super(Tox21GlobalMeanPool, self).__init__()

    def forward(self, value):
        x, batch = value.x, value.data.batch
        x = global_mean_pool(x, batch)
        value.x = x        
        return value

class Tox21Linear(nn.Module):
    def __init__(self, out_channels, batch_size, activation=True):
        super(Tox21Linear, self).__init__()
        self.batch_size = batch_size
        self.pool = Tox21GlobalMeanPool()        
        self.out_channels = out_channels        
        self.linear = exnn.Linear(out_channels)
        self.relu = nn.ReLU()
        self.activation = activation

    def forward(self, value):
        batch_dim_size = value.x.size(0)
        if batch_dim_size != self.batch_size:
            value = self.pool(value)
        x = value.x
        x = self.linear(x)
        if self.activation:
            x = self.relu(x)
        value.x = x                
        return value

class Tox21FlattenLinear(nn.Module):
    def __init__(self, out_channels, batch_size):
        super(Tox21FlattenLinear, self).__init__()
        self.batch_size = batch_size
        self.pool = Tox21GlobalMeanPool()
        self.out_channels = out_channels
        self.linear = exnn.Linear(self.out_channels)
        self.relu = nn.ReLU()

    def forward(self, value):
        batch_dim_size = value.x.size(0)        
        if batch_dim_size != self.batch_size:
            x = self.pool(value)
        x = value.x
        x = self.relu(x)
        if len(x.shape) == 4:
            B, _, _, _ = x.shape
            x = x.reshape(B, -1)
        value.x = x                            
        return value

class Tox21ReLU(nn.Module):
    def __init__(self):
        super(Tox21ReLU, self).__init__()

    def forward(self, value):
        x = value.x
        x = F.relu(x)
        value.x = x                                    
        return value
    
class Tox21ConcatFlatten(nn.Module):
    def __init__(self, out_channels, batch_size):
        super(Tox21ConcatFlatten, self).__init__()
        self.out_channels = out_channels
        self.batch_size = batch_size
        self.linear = exnn.Linear(self.out_channels)
        self.relu = nn.ReLU()
        self.pool = Tox21GlobalMeanPool()        
        
    def forward(self, *values):
        x1 = values[0].x
        x2 = values[1].x
        x = torch.cat([x1, x2], dim=1)
        if len(x1.shape) == 4:
            B, _, _, _ = x.shape
            x = x.reshape(B, -1)
        x = self.relu(self.linear(x))
        values[0].x = x                                    
        return values[0]

class Tox21Concat(nn.Module):
    def __init__(self, batch_size):
        super(Tox21Concat, self).__init__()
        self.pool = Tox21GlobalMeanPool()
        self.batch_size = batch_size
        
    def forward(self, *values):
        if len(values) == 1:
            return values
        _values = []
        for v in values:
            if v.x.size(0) != self.batch_size:
                v = self.pool(v)
            _values.append(v.x)
        x = torch.cat(_values, dim=1)
        values[0].x = x                                    
        return values[0]

    
class Tox21ConcatLinear(nn.Module):
    def __init__(self, out_channels):
        super(Tox21ConcatLinear, self).__init__()
        self.pool = Tox21GlobalMeanPool()
        self.linear = exnn.Linear(out_channels)
        
    def forward(self, *values):
        if len(values) == 1:
            x = self.linear(values[0].x)
            values[0].x = x
            return values[0]
        _values = []
        for v in values:
            if v.x.size(0) != self.batch_size:
                v = self.pool(v)
            _values.append(v.x)
        x = torch.cat(_values, dim=1)
        x = self.linear(x)        
        values[0].x = x
        return values[0]
    

class Tox21GCNorLinear(nn.Module):
    def __init__(self, out_channels, batch_size):
        super(Tox21GCNorLinear, self).__init__()
        self.batch_size = batch_size
        self.linear = exnn.Linear(out_channels)
        self.gcn = LazyGCNConv(out_channels)
        self.relu = nn.ReLU()

    def forward(self, value):
        batch_dim_size = value.x.size(0)
        x = value.x
        if batch_dim_size != self.batch_size:
            adj, edge_index = x.float(), value.data.edge_index
            x = self.gcn(adj, edge_index)
        else:
            x = self.linear(x)                    
        x = self.relu(x)            
        value.x = x                                                
        return value

if __name__ == '__main__':
    from kgcn_torch.datasets.tox21 import Tox21Dataset
    from torch_geometric.data import DataLoader
    train_loader = DataLoader(Tox21Dataset("train"), batch_size=16, shuffle=True)
    for batch_idx, data in enumerate(train_loader):
        x = (data.x.float(), data)
        print(torch.max(data.batch))
        print(data.x)
        print(data.edge_index)
        print(data.y)
        print(o)
        a
        
