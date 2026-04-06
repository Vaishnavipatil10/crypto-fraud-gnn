import torch

def predict_gnn(model, x, edge_index):
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
        return out.argmax(dim=1)

def predict_lstm(model, seq):
    model.eval()
    with torch.no_grad():
        return model(seq)
