"""
Yield Predictor
Predicts Yield (%) from process operating conditions.
"""

import streamlit as st
import pandas as pd
import numpy as np

import plotly.graph_objects as go

# ----------------------------------------
# Page Configuration
# ----------------------------------------

st.set_page_config(
    page_title="Yield Predictor",
    page_icon="⚗️",
    layout="wide"
)
# ----------------------------------------
# Load Trained Model
# ----------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("yield_model.joblib")

bundle = load_model()

model = bundle["model"]
FEATURES = bundle["features"]
RANGES = bundle["feat_ranges"]
CATEGORIES = bundle["categories"]

Y_MEAN = bundle["y_mean"]
CV_R2 = bundle["cv_r2"]
CV_MAE = bundle["cv_mae"]
BEST_MODEL = bundle["best_model"]

# ----------------------------------------
# Title
# ----------------------------------------

st.title("⚗️ Yield Predictor")

st.markdown(
    """
Predict the **Yield (%)** using the reactor operating conditions.

Enter the operating conditions in the left panel.
The trained machine learning model will estimate the expected yield.
"""
)
# ----------------------------------------
# Input Controls
# ----------------------------------------

left, right = st.columns([1, 1.2])

with left:

    st.header("Operating Conditions")

    values = {}

    # Temperature
    lo, hi, med = RANGES["Temperature (°C)"]
    values["Temperature (°C)"] = st.slider(
        "Temperature (°C)",
        float(lo),
        float(hi),
        float(med)
    )

    # Pressure
    lo, hi, med = RANGES["Pressure (bar)"]
    values["Pressure (bar)"] = st.slider(
        "Pressure (bar)",
        float(lo),
        float(hi),
        float(med)
    )

    # Concentration
    lo, hi, med = RANGES["Concentration (mol/L)"]
    values["Concentration (mol/L)"] = st.slider(
        "Concentration (mol/L)",
        float(lo),
        float(hi),
        float(med)
    )

    # Catalyst
    lo, hi, med = RANGES["Catalyst (wt%)"]
    values["Catalyst (wt%)"] = st.slider(
        "Catalyst (wt%)",
        float(lo),
        float(hi),
        float(med)
    )

    # Residence Time
    lo, hi, med = RANGES["Residence Time (min)"]
    values["Residence Time (min)"] = st.slider(
        "Residence Time (min)",
        float(lo),
        float(hi),
        float(med)
    )

    # Reactor Type
    values["Reactor Type"] = st.selectbox(
        "Reactor Type",
        CATEGORIES["Reactor Type"]
    )
    # ----------------------------------------
# Prediction
# ----------------------------------------

input_df = pd.DataFrame([values])

prediction = model.predict(input_df)[0]

prediction = np.clip(prediction, 0, 100)

with right:

    st.header("Predicted Yield")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prediction,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "green"},
            },
        )
    )

    fig.update_layout(height=350)

    st.plotly_chart(fig, use_container_width=True)

    st.metric(
        "Predicted Yield (%)",
        f"{prediction:.2f}%"
    )
