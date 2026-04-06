import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_price_data():
    df = pd.read_csv("data/crypto_price.csv")
    df = df[['Close']]
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    
    return scaled_data, scaler

def create_sequences(data, seq_length=10):
    X, y = [], []
    
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    
    return np.array(X), np.array(y)
