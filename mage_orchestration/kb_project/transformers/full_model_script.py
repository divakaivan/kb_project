import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score

import mlflow
from torchsummary import summary

mlflow.set_tracking_uri('http://mlflow-server:5001')
mlflow.set_experiment('kb_project')

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

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

def compute_time_difference(group):
    n = len(group)
    result = []
    for i in range(n):
        for j in range(n):
            time_difference = abs(group.iloc[i].trans_date_trans_time.value - group.iloc[j].trans_date_trans_time.value)
            result.append([group.iloc[i].name, group.iloc[j].name, time_difference])
    return result

def create_folder(name):
    date_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"models_{name}_{date_now}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name

@transformer
def transform(data, *args, **kwargs):
    credit_card_nodes, merchant_nodes, transactions = data
    all_df = transactions.join(credit_card_nodes.set_index('cc_num'), on='cc_num').join(merchant_nodes.set_index('merchant'), on='merchant')
    all_df = all_df.assign(trans_date_trans_time= list(map(lambda x: pd.to_datetime(x), all_df.trans_date_trans_time)))

    df_not_fraud = all_df[all_df['is_fraud'] == 0].sample(frac=0.2, random_state=999)
    df_fraud = all_df[all_df['is_fraud'] == 1]
    df_undersampled = pd.concat([df_not_fraud, df_fraud])

    non_fraud_rows = df_undersampled[df_undersampled['is_fraud'] == 0].copy()
    fraud_rows = df_undersampled[df_undersampled['is_fraud'] == 1].copy()
    non_fraud_downsampled = resample(
            non_fraud_rows, 
            n_samples=len(fraud_rows), 
            replace=False, 
            random_state=42
        )
    balanced_df = pd.concat([fraud_rows, non_fraud_downsampled])
    balanced_df = balanced_df.reset_index(drop=True)

    balanced_df_train, balanced_df_test = train_test_split(balanced_df, test_size=0.25, random_state=42)

    n = len(balanced_df)

    train_mask = [i in balanced_df_train.index for i in range(n)]
    test_mask = [i in balanced_df_test.index for i in range(n)]
    train_mask = np.array(train_mask)
    test_mask = np.array(test_mask)

    if os.path.exists('datasets/edge_index.npy'):
        print('edge_index.npy exists. Using it')
        edge_index = np.load('datasets/'+'edge_index.npy').astype(np.float64)
    else:
        print('edge_index.npy does not exist. Creating it')
        groups = balanced_df.groupby('cc_num')
        edge_index_list = [compute_time_difference(group) for _, group in groups]
        edge_index_list_flat = [item for sublist in edge_index_list for item in sublist]
        edge_index = np.array(edge_index_list_flat).astype(np.float64)
        np.save('datasets/edge_index.npy', edge_index)

    theta = edge_index[:,2].mean()
    edge_index[:,2] = (np.exp(-edge_index[:,2]/theta) != 1)*(np.exp(-edge_index[:,2]/theta))
    edge_index = edge_index.tolist()
    mean_ = np.array(edge_index)[:,2].mean()

    selected_edges = [(int(row[0]), int(row[1])) for row in edge_index if row[2] > mean_]
    edge_index_selected = torch.tensor(selected_edges, dtype=torch.long).t()
    
    category_dummies = pd.get_dummies(balanced_df['category'], drop_first=True, prefix='cat')
    balanced_df = pd.concat([balanced_df, category_dummies], axis=1)

    features = balanced_df[['lat', 'long', 'amt', 'merch_lat', 'merch_long', *category_dummies.columns]].values

    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(balanced_df['is_fraud'],dtype=torch.int64)
    data = Data(x=x, edge_index = edge_index_selected, y=y, train_mask = train_mask, test_mask = test_mask)
    print(f'{data=}')

    y_test = (data.y[data.test_mask]).numpy()

    losses = []
    
    with mlflow.start_run():
        
        model = GCN()
        lr = 0.003
        weight_decay = 5e-4
        mlflow.log_param('lr', lr)
        mlflow.log_param('weight_decay', weight_decay)

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=5e-4)
        model.train()
        
        for epoch in range(101):
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            losses.append(loss.item())
            if epoch % 20 == 0:
                print(f'Epoch: {epoch}, Loss: {loss.item()}')
            loss.backward()
            optimizer.step()

        # Save model
        save_path = create_folder('gcn')
        model_path = f'{save_path}/gcn.pth'
        torch.save(model.state_dict(), model_path)
        print('Model saved!')

        # Log model to MLflow
        mlflow.log_artifact(model_path, "model")

        model.eval()

        with torch.no_grad():
            out = model(data.x, data.edge_index).argmax(dim=1)
            y_pred = out[data.test_mask]

        accuracy = round(accuracy_score(y_test, y_pred), 4)
        precision = round(precision_score(y_test, y_pred), 4)
        recall = round(recall_score(y_test, y_pred), 4)
        f1 = round(f1_score(y_test, y_pred), 4)

        result_string = f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}'
        print(result_string)

        # Log metrics to MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        results_path = os.path.join(save_path, 'metrics.txt')
        with open(results_path, 'w') as file:
            file.write(result_string)
        mlflow.log_artifact(results_path)

        # Log model architecture to MLflow
        model_summary_str = str(summary(model))
        summary_path = os.path.join(save_path, 'model_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(model_summary_str)
        mlflow.log_artifact(summary_path)

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Fraud', 'Fraud'])
        disp.plot()

        # Save and log confusion matrix
        cm_image_path = f'{save_path}/conf_matrix.png'
        plt.savefig(cm_image_path)
        mlflow.log_artifact(cm_image_path)

        plt.show()

        # Explainability
        model.eval()
        from torch_geometric.explain import Explainer, GNNExplainer
        explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=200),
            explanation_type='model',
            node_mask_type='common_attributes',
            model_config=dict(
                mode='binary_classification',
                task_level='node',
                return_type='raw'
            )
        )
        
        node_index = 10
        explanation = explainer(data.x, data.edge_index)
        print(f'Generated explanations in {explanation.available_explanations}')

        feat_imp_path = f'{save_path}/feature_importance.png'
        explanation.visualize_feature_importance(feat_imp_path, feat_labels=['lat', 'long', 'amt', 'merch_lat', 'merch_long', *category_dummies.columns])
        mlflow.log_artifact(feat_imp_path)
        print(f"Feature importance plot has been saved to '{feat_imp_path}'")
 