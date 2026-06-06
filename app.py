import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.express as px
import os

# ==================================================
# PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND)
# ==================================================

st.set_page_config(
    page_title="Fraud Intelligence Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.metric-container {
    background: #111827;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #374151;
}

.big-title {
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:#60A5FA;
}

.subtitle {
    text-align:center;
    font-size:18px;
    color:#9CA3AF;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# CUSTOM POSITIONAL ENCODING
# ==================================================

@tf.keras.utils.register_keras_serializable()
class PositionalEncoding(tf.keras.layers.Layer):

    def call(self, x):
        seq_len = tf.shape(x)[1]

        positions = tf.range(
            start=0,
            limit=seq_len,
            delta=1
        )

        positions = tf.cast(
            positions,
            tf.float32
        )

        return x + tf.expand_dims(
            positions,
            axis=-1
        )

    def get_config(self):
        return super().get_config()

# ==================================================
# LOAD MODEL
# ==================================================

@st.cache_resource
def load_model():

    try:

        model = tf.keras.models.load_model(
            "attention_model.keras",
            custom_objects={
                "PositionalEncoding": PositionalEncoding
            },
            compile=False
        )

        return model

    except Exception as e:

        st.error("Failed to load attention_model.keras")
        st.exception(e)
        st.stop()

@st.cache_resource
def load_scaler():

    try:

        if os.path.exists("scaler.pkl"):
            return joblib.load("scaler.pkl")

        elif os.path.exists("scaler .pkl"):
            return joblib.load("scaler .pkl")

        else:
            st.error("Scaler file not found.")
            st.stop()

    except Exception as e:

        st.error("Failed to load scaler.")
        st.exception(e)
        st.stop()

model = load_model()
scaler = load_scaler()

# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.title("🛡️ Fraud AI")

    st.markdown("---")

    st.markdown("""
### Features

✅ Transaction Analysis

✅ Fraud Detection

✅ Risk Scoring

✅ Interactive Dashboard

✅ CSV Export
""")

    st.markdown("---")

    st.info(
        "Model: LSTM + Attention + Positional Encoding"
    )

# ==================================================
# HEADER
# ==================================================

st.markdown(
    "<div class='big-title'>💳 Fraud Intelligence Command Center</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Deep Learning Based Fraud Detection Dashboard</div>",
    unsafe_allow_html=True
)

st.markdown("---")

# ==================================================
# FILE UPLOAD
# ==================================================

st.subheader("📂 Upload Transaction Dataset")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

# ==================================================
# PROCESS FILE
# ==================================================

if uploaded_file is not None:

    try:

        df = pd.read_csv(uploaded_file)

        st.success("Dataset uploaded successfully.")

        col1, col2 = st.columns([3, 1])

        with col1:
            st.subheader("Dataset Preview")
            st.dataframe(
                df.head(),
                use_container_width=True
            )

        with col2:
            st.subheader("Dataset Info")

            st.metric(
                "Rows",
                df.shape[0]
            )

            st.metric(
                "Columns",
                df.shape[1]
            )

        st.markdown("---")

        if st.button(
            "🚀 Analyze Transactions",
            use_container_width=True
        ):

            with st.spinner(
                "Running fraud detection model..."
            ):

                # ====================================
                # ORIGINAL LOGIC
                # ====================================

                working_df = df.copy()

                if "Class" in working_df.columns:

                    features_df = working_df.drop(
                        columns=["Class"]
                    )

                else:

                    features_df = working_df

                feature_values = features_df.values

                scaled_features = scaler.transform(
                    feature_values
                )

                SEQ_LEN = 5

                sequences = []

                for i in range(
                    len(scaled_features) - SEQ_LEN
                ):

                    sequences.append(
                        scaled_features[
                            i:i + SEQ_LEN
                        ]
                    )

                X = np.array(sequences)

                if len(X) == 0:

                    st.error(
                        "Dataset must contain at least 6 rows."
                    )

                    st.stop()

                predictions = model.predict(
                    X,
                    verbose=0
                )

                fraud_prob = predictions.flatten()

                results = df.iloc[
                    SEQ_LEN:
                ].copy()

                results[
                    "Fraud_Probability"
                ] = fraud_prob

                avg_prob = fraud_prob.mean() * 100

                high_risk = results[
                    results[
                        "Fraud_Probability"
                    ] > 0.80
                ]

                # ====================================
                # KPI SECTION
                # ====================================

                st.markdown("## 📊 Executive Summary")

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.metric(
                        "Transactions Analyzed",
                        len(results)
                    )

                with c2:
                    st.metric(
                        "Average Fraud Risk",
                        f"{avg_prob:.2f}%"
                    )

                with c3:
                    st.metric(
                        "High Risk Transactions",
                        len(high_risk)
                    )

                st.progress(
                    min(avg_prob / 100, 1.0)
                )

                # ====================================
                # TABS
                # ====================================

                tab1, tab2, tab3 = st.tabs(
                    [
                        "🚨 Detection Results",
                        "📈 Analytics",
                        "⬇️ Export"
                    ]
                )

                # =============================
                # RESULTS TAB
                # =============================

                with tab1:

                    st.subheader(
                        "High Risk Transactions"
                    )

                    if len(high_risk) > 0:

                        st.error(
                            f"{len(high_risk)} high-risk transactions detected."
                        )

                        st.dataframe(
                            high_risk,
                            use_container_width=True
                        )

                    else:

                        st.success(
                            "No high-risk transactions detected."
                        )

                    st.subheader(
                        "Top 10 Riskiest Transactions"
                    )

                    top10 = results.sort_values(
                        by="Fraud_Probability",
                        ascending=False
                    ).head(10)

                    st.dataframe(
                        top10,
                        use_container_width=True
                    )

                # =============================
                # ANALYTICS TAB
                # =============================

                with tab2:

                    st.subheader(
                        "Fraud Probability Distribution"
                    )

                    fig = px.histogram(
                        results,
                        x="Fraud_Probability",
                        nbins=30,
                        title="Fraud Probability Distribution"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                    st.subheader(
                        "Fraud Score Spread"
                    )

                    fig2 = px.box(
                        results,
                        y="Fraud_Probability",
                        title="Fraud Score Spread"
                    )

                    st.plotly_chart(
                        fig2,
                        use_container_width=True
                    )

                # =============================
                # EXPORT TAB
                # =============================

                with tab3:

                    st.subheader(
                        "Download Prediction Results"
                    )

                    csv = results.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="📥 Download Predictions CSV",
                        data=csv,
                        file_name="fraud_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                    st.dataframe(
                        results.head(20),
                        use_container_width=True
                    )

    except Exception as e:

        st.error(
            f"Prediction Error: {str(e)}"
        )

        st.exception(e)
