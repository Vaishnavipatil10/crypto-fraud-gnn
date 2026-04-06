import shap
import torch

########################################
# SHAP FOR LSTM
########################################
def shap_lstm(model, X):
    def predict_fn(data):
        data = torch.tensor(data, dtype=torch.float32)
        return model(data).detach().numpy()

    explainer = shap.KernelExplainer(predict_fn, X[:50].numpy())
    shap_values = explainer.shap_values(X[:10].numpy())

    return shap_values


########################################
# SHAP FOR GNN (Approx)
########################################
def shap_gnn(model, x, edge_index):
    def predict_fn(data):
        data = torch.tensor(data, dtype=torch.float32)
        return model(data, edge_index).detach().numpy()

    explainer = shap.KernelExplainer(predict_fn, x[:100].numpy())
    shap_values = explainer.shap_values(x[:10].numpy())

    return shap_values


########################################
# Gradient Explainability
########################################
def explain_node(model, x, edge_index, node_idx=0):
    x.requires_grad = True
    out = model(x, edge_index)
    pred = out[node_idx].argmax()

    out[node_idx, pred].backward()
    importance = x.grad[node_idx].abs()

    return importance.detach().numpy()
