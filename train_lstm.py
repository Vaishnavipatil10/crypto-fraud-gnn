import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from utils.model.lstm_model import LSTMModel

df = pd.read_csv("data/crypto_price.csv")
prices = df['price'].values.reshape(-1, 1)

scaler = MinMaxScaler()
prices = scaler.fit_transform(prices)

def create_seq(data, seq_len=10):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

X, y = create_seq(prices)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

model = LSTMModel()
opt = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

for epoch in range(30):
    opt.zero_grad()
    out = model(X)
    loss = loss_fn(out, y)

    loss.backward()
    opt.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")

torch.save(model.state_dict(), "lstm_model.pth")
