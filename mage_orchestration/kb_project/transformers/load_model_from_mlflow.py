import mlflow

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from typing import Dict, List
from datetime import datetime

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(18, 64)
        self.conv2 = GCNConv(64,32)
        self.conv3 = GCNConv(32,16)
        self.conv4 = GCNConv(16,2)

    def forward(self, x, edge_index):
        # x, edge_index = data.x, data.edge_index

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

model = GCN()
model_path = 'models_20240728_050159/gcn.pth' # can replace with mlflow if setup
model.load_state_dict(torch.load(model_path))
model.eval()

@transformer
def transform(messages: List[Dict], *args, **kwargs):

    for msg in messages:
        numerical_cols = ['lat', 'long', 'amt', 'merch_lat', 'merch_long']
        categories = {
        'food_dining': 0, 'gas_transport': 0, 'grocery_net': 0, 'grocery_pos': 0,
        'health_fitness': 0, 'home': 0, 'kids_pets': 0, 'misc_net': 0, 'misc_pos': 0,
        'personal_care': 0, 'shopping_net': 0, 'shopping_pos': 0, 'travel': 0
        }

        category = msg.get('category')
        try:
            categories[category] += 1
        except KeyError:
            pass
        numericals_data = [msg.get(num) for num in numerical_cols]
        data_inputs = numericals_data + list(categories.values())
        data_inputs = torch.tensor(data_inputs, dtype=torch.float32).unsqueeze(0)
        empty_edge_index = torch.empty((2, 0), dtype=torch.long) 
        single_test_data = Data(x=data_inputs, edge_index=empty_edge_index)
        with torch.no_grad():
            prediction = model(single_test_data.x, single_test_data.edge_index)

        predicted_class = prediction.argmax(dim=1).item()
        msg['pred_gcn_is_fraud'] = predicted_class

        # fake date as if transactions are recent
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        msg['trans_date_trans_time'] = now
    
    return messages

