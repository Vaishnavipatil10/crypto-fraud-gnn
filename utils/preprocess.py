import pandas as pd
import torch
from torch_geometric.data import Data

def load_data():
    features = pd.read_csv("data/elliptic_txs_features.csv", header=None)
    edges = pd.read_csv("data/elliptic_txs_edgelist.csv")
    classes = pd.read_csv("data/elliptic_txs_classes.csv")

    x = torch.tensor(features.iloc[:, 1:].values, dtype=torch.float)
    edge_index = torch.tensor(edges.values.T, dtype=torch.long)

    y = classes['class'].replace({'unknown': 2})
    y = torch.tensor(y.values, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)