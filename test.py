import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)

MODELS_DIR = Path("models")  # update this if your models folder is elsewhere
assert MODELS_DIR.exists(), f"Models folder not found at {MODELS_DIR.resolve()}"

import base64
from pathlib import Path


def set_bg(image_file):
    with open(image_file, "rb") as f:
        data = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                linear-gradient(
                    rgba(0,0,0,0.7),
                    rgba(0,0,0,0.9)
                ),
                url("data:image/png;base64,{data}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

BASE_DIR = Path(__file__).resolve().parent
set_bg(BASE_DIR / "pic 2.jpg")
# ----------------------------------------------------------------------------
# Page config & styling
# ----------------------------------------------------------------------------

st.set_page_config(
    page_title="Diamond Intelligences Studio",
    page_icon="💎",
    layout="wide",
)
st.markdown(
    """
    <style>
         .stApp { background-color: #0f1117; }
        .main-title {
            font-size: 2.4rem;
            font-weight: 700;
            background: linear-gradient(90deg, #b993f4, #8ac6ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0rem;
        }
        .subtitle { color: #9aa4b2; font-size: 1.05rem; margin-top: 0; }
        .result-card {
            padding: 1.4rem 1.6rem;
            border-radius: 14px;
            background: linear-gradient(135deg, rgba(185,147,244,0.12), rgba(138,198,255,0.08));
            border: 1px solid rgba(185,147,244,0.35);
            margin-top: 0.6rem;
        }
        .result-label { color: #9aa4b2; font-size: 0.95rem; margin-bottom: 0.2rem; }
        .result-value { font-size: 2.1rem; font-weight: 700; color: #f5f5f7; }
        .cluster-badge {
            display: inline-block;
            padding: 0.35rem 0.9rem;
            border-radius: 999px;
            background: rgba(138,198,255,0.15);
            border: 1px solid rgba(138,198,255,0.4);
            color: #8ac6ff;
            font-weight: 600;
            font-size: 0.95rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Load artifacts
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    
    with open(MODELS_DIR / "xgb_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    with open(MODELS_DIR / "feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)
    with open(MODELS_DIR / "ordinal_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    with open(MODELS_DIR / "standard_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(MODELS_DIR / "power_transformer.pkl", "rb") as f:
        power_transformer = pickle.load(f)
    with open(MODELS_DIR / "pca.pkl", "rb") as f:
        pca = pickle.load(f)
    with open(MODELS_DIR / "clustering_model.pkl", "rb") as f:
        kmeans = pickle.load(f)
    with open(MODELS_DIR / "cluster_names.pkl", "rb") as f:
        cluster_names = pickle.load(f)
    return xgb_model, feature_columns, encoder , scaler, power_transformer, pca, kmeans, cluster_names

(
    xgb_model,
    FEATURE_COLUMNS,
    encoder,
    scaler,
    power_transformer,
    pca,
    kmeans,
    cluster_names,
) = load_artifacts()


POWER_COLS = list(power_transformer.feature_names_in_)

# ----------------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------------
def build_feature_row(carat, cut, color, clarity, length, width, depth, table):
    
    cat_df = pd.DataFrame(
    [[cut, color, clarity]],
    columns=["cut", "color", "clarity"]    )
    cut_enc, color_enc, clarity_enc = encoder.transform(cat_df)[0]

    depth_pct = depth / ((length + width) / 2) * 100
    dimension_ratio = (length * width) / (depth * 2)
    volume = length * width * depth
    face_area = length * width

    row = {
        "carat": carat,
        "cut": cut_enc,
        "color": color_enc,
        "clarity": clarity_enc,
        "depth_pct": depth_pct,
        "table": table,
        "length": length,
        "width": width,
        "depth": depth,
        "dimension_ratio": dimension_ratio,
        "volume": volume,
        "face_area": face_area,
    }
    df = pd.DataFrame([row])[FEATURE_COLUMNS]
    return df
    
def predict_price(raw_df):
    # XGB model uses raw (unscaled) engineered features directly
    pred = xgb_model.predict(raw_df)[0]
    return float(pred)

def predict_cluster(raw_df):
    df = raw_df.copy()
    # 1. Power-transform the skewed subset of columns
    df["table"] = np.sqrt(df["table"])
    df[POWER_COLS] = power_transformer.transform(df[POWER_COLS])
    # 2. Standard-scale all 12 columns (same order the scaler was fit on)
    scaled = scaler.transform(df[FEATURE_COLUMNS])
    # 3. PCA
    pca_vec = pca.transform(scaled)
    # 4. KMeans
    cluster_id = int(kmeans.predict(pca_vec)[0])
    cluster_label = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
    
    # return cluster_id, cluster_label.replace("_", " "), pca_vec
    return (cluster_id,cluster_label.replace("_", " "),
    pca_vec,df,scaled,kmeans.transform(pca_vec))
    st.write("Raw Features")
    st.write(raw_df)
    
    st.write("Processed Features")
    st.write(df)
    
    st.write("PCA Vector")
    st.write(pca_vec)
    
    st.write("Distances")
    st.write(kmeans.transform(pca_vec))

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown('<p class="main-title">💎 Diamond Intelligence Studio</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Predict diamond price and discover its market segment using trained ML models.</p>',
    unsafe_allow_html=True,
)
st.write("")

# ----------------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------------
with st.container():
    st.markdown("### 📝 Diamond Attributes")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        carat = st.number_input("Carat", min_value=0.01, max_value=10.0, value=0.70, step=0.01, format="%.2f")
    with c2:
        length = st.number_input("Length — x (mm)", min_value=0.1, max_value=15.0, value=5.70, step=0.01, format="%.2f")
    with c3:
        width = st.number_input("Width — y (mm)", min_value=0.1, max_value=15.0, value=5.72, step=0.01, format="%.2f")
    with c4:
        depth = st.number_input("Depth — z (mm)", min_value=0.1, max_value=15.0, value=3.53, step=0.01, format="%.2f")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        table = st.number_input("Table (%)", min_value=40.0, max_value=80.0, value=57.5, step=0.1, format="%.1f")
    with c6:
        cut = st.selectbox("Cut", list(encoder.categories_[0]))
    with c7:
        color = st.selectbox("color", list(encoder.categories_[1]))
    with c8:
        clarity = st.selectbox("clarity", list(encoder.categories_[2]))

st.write("")

# Build the shared feature row once, from the current form state
raw_df = build_feature_row(carat, cut, color, clarity, length, width, depth, table)

# ----------------------------------------------------------------------------
# Action buttons
# ----------------------------------------------------------------------------
b1, b2 = st.columns(2)
with b1:
    price_clicked = st.button("💰 Predict Price", use_container_width=True, type="primary")
with b2:
    cluster_clicked = st.button("📊 Predict Cluster", use_container_width=True)

st.write("")

# ----------------------------------------------------------------------------
# Price prediction output
# ----------------------------------------------------------------------------

if price_clicked:
    price = predict_price(raw_df)
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Predicted Diamond Price</div>
            <div class="result-value">₹ {price:,.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------------
# Cluster prediction output
# ----------------------------------------------------------------------------
if cluster_clicked:
    # cluster_id, cluster_label, pca_vec = predict_cluster(raw_df)
    cluster_id, cluster_label, pca_vec, processed_df, scaled, distances = predict_cluster(raw_df)   


    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Market Segment</div>
            <div class="result-value">Cluster {cluster_id}</div>
            <div style="margin-top:0.5rem;"><span class="cluster-badge">{cluster_label}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )




