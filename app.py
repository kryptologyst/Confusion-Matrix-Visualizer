import streamlit as st
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data import load_data
from src.model import ConfusionMatrixTool

st.set_page_config(page_title="Confusion Matrix", page_icon="📋", layout="wide")
st.title("📋 Confusion Matrix Visualizer")
st.markdown("Train a classifier and visualize its confusion matrix.")

dataset_name = st.selectbox("Dataset", ["wine", "iris", "breast_cancer"])
X, y, fn, cn = load_data(dataset_name)

normalize = st.checkbox("Normalize")
if st.button("Generate", type="primary"):
    tool = ConfusionMatrixTool()
    results = tool.train_and_evaluate(X, y, class_names=cn)
    st.metric("Accuracy", f"{results['metrics']['accuracy']:.2%}")
    cm = np.array(results["confusion_matrix"])
    if normalize:
        cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        import plotly.express as px
        fig = px.imshow(cm_norm, x=cn, y=cn, text_auto=".2f", color_continuous_scale="Blues", zmin=0, zmax=1)
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        import plotly.express as px
        fig = px.imshow(cm, x=cn, y=cn, text_auto=True, color_continuous_scale="Blues")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
