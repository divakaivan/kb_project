import numpy as np

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(18, 64)
        self.conv2 = GCNConv(64,32)
        self.conv3 = GCNConv(32,16)
        self.conv4 = GCNConv(16,2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv4(x, edge_index)

        return F.log_softmax(x, dim=1)

@custom
def transform_custom(data, *args, **kwargs):
    X_train, X_test, y_train, y_test = data
    X_train = X_train_df.values
    X_test = X_test_df.values
    y_train = y_train_s.values
    y_test = y_test_s.values


    

@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
