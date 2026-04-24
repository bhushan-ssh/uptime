import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ---- Page config (FIXED to wide) ----
st.set_page_config(page_title="Predictive Maintenance", layout="wide")

# ---- Title ----
st.markdown("""
<style>
.big-font {
    font-size:32px !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">🔧 AI Predictive Maintenance</p>', unsafe_allow_html=True)
st.caption("Predict Remaining Useful Life (RUL) from recent sensor history")

# ---- Load model ----
model = tf.keras.models.load_model("models/lstm_model.h5", compile=False)
scaler = joblib.load("models/scaler.pkl")

# ---- Model shape ----
_, seq_len, n_features = model.input_shape

# ---- Sidebar ----
st.sidebar.title("⚙️ Controls")
use_random = st.sidebar.checkbox("Use random demo input", value=True)

# ---- Input ----
st.subheader(f"Input (last {seq_len} cycles, {n_features} features)")

if use_random:
    X_input = np.random.rand(1, seq_len, n_features)
else:
    st.info(f"Upload CSV with shape ({seq_len}, {n_features})")
    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)

        if df.shape != (seq_len, n_features):
            st.error(f"❌ Expected shape ({seq_len}, {n_features}), got {df.shape}")
            X_input = None
        else:
            df_scaled = scaler.transform(df)
            X_input = df_scaled.reshape(1, seq_len, n_features)
    else:
        X_input = None

# ---- Status function ----
def get_status(rul):
    if rul > 50:
        return "🟢 SAFE", "green"
    elif rul > 20:
        return "🟡 WARNING", "orange"
    else:
        return "🔴 CRITICAL", "red"

# ---- Predict ----
if st.button("Predict RUL") and X_input is not None:

    pred = float(model.predict(X_input)[0][0])
    label, color = get_status(pred)

    # ---- Layout columns ----
    col1, col2 = st.columns(2)

    # LEFT: Results
    with col1:
        st.subheader("📊 Prediction Result")
        st.metric("Predicted RUL", round(pred, 2))
        st.markdown(f"### Status: :{color}[{label}]")

        # Health bar
        health = max(0, min(100, int(pred)))
        st.progress(health)
        st.caption("Engine Health (Higher = Better)")

        # Smart message
        if pred > 50:
            st.success("Engine is operating normally. No immediate action required.")
        elif pred > 20:
            st.warning("Maintenance should be scheduled soon to avoid failure.")
        else:
            st.error("Immediate maintenance required! High risk of failure.")

    # RIGHT: Gauge
    with col2:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred,
            title={'text': "RUL Score"},
            gauge={
                'axis': {'range': [0, 125]},
                'bar': {'color': "blue"},
                'steps': [
                    {'range': [0, 20], 'color': "red"},
                    {'range': [20, 50], 'color': "orange"},
                    {'range': [50, 125], 'color': "green"},
                ],
            }
        ))

        st.plotly_chart(gauge, use_container_width=True)

    # ---- Graph ----
    st.subheader("📈 Sensor Trends")

    fig, ax = plt.subplots()

    for i in range(min(5, n_features)):
        ax.plot(X_input[0][:, i], label=f'Feature {i+1}')

    ax.set_title("Key Sensor Trends")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Normalized Value")
    ax.legend()

    st.pyplot(fig, use_container_width=True)

# ---- Footer ----
st.markdown("---")
st.caption("Built using LSTM on NASA C-MAPSS Dataset | Predictive Maintenance System")