import streamlit as st

st.title("🚀 Crypto Project Debug Mode")

st.write("App started successfully ✅")

# -------- DEBUG CHECK 1 --------
try:
    import torch
    st.write("✅ Torch imported")
except Exception as e:
    st.error(f"Torch error: {e}")

# -------- DEBUG CHECK 2 --------
try:
    from utils.preprocess import load_data
    data = load_data()
    st.write("✅ Data loaded")
except Exception as e:
    st.error(f"Data error: {e}")

# -------- DEBUG CHECK 3 --------
try:
    from model.gnn_model import GNN
    model = GNN(data.num_features)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    st.write("✅ Model loaded")
except Exception as e:
    st.error(f"Model error: {e}")

# -------- BUTTON --------
if st.button("Run Test"):
    st.write("Button working ✅")