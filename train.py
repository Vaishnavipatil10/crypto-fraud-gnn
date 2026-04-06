import torch
from torch_geometric.data import Data
from utils.model.gnn_model import GNNModel
from utils.preprocess import load_graph_data

X, edges, y = load_graph_data()

x = torch.tensor(X, dtype=torch.float32)
edge_index = torch.tensor(edges.T, dtype=torch.long)
y = torch.tensor(y, dtype=torch.long)

data = Data(x=x, edge_index=edge_index, y=y)

model = GNNModel(x.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(30):
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)
    loss = torch.nn.functional.nll_loss(out[y != -1], y[y != -1])

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")

torch.save(model.state_dict(), "gnn_model.pth")
