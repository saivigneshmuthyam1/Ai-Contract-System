# ==================================================
# PAGE CONFIG
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
    background-color: #0E1117;
}

.metric-card {
    background: linear-gradient(135deg,#1E293B,#0F172A);
    padding:20px;
    border-radius:15px;
    border:1px solid #334155;
    text-align:center;
}

.block-container {
    padding-top: 1rem;
}

.kpi {
    background:#111827;
    padding:20px;
    border-radius:15px;
    border-left:5px solid #3B82F6;
}

.success-box {
    background:#052e16;
    padding:15px;
    border-radius:10px;
}

.warning-box {
    background:#3f1d1d;
    padding:15px;
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/3064/3064197.png",
        width=120
    )

    st.title("Fraud AI")

    st.markdown("---")

    st.markdown("""
    ### Dashboard Modules

    ✅ Upload Data

    ✅ Risk Analysis

    ✅ Fraud Detection

    ✅ Visual Analytics

    ✅ Export Results
    """)

    st.markdown("---")

    st.info(
        "LSTM + Attention + Positional Encoding Model"
    )

# ==================================================
# HERO SECTION
# ==================================================

st.markdown("""
<h1 style='text-align:center;color:#60A5FA'>
🛡️ Fraud Intelligence Command Center
</h1>

<p style='text-align:center;font-size:18px'>
Advanced Deep Learning Based Fraud Detection System
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ==================================================
# FILE UPLOAD SECTION
# ==================================================

st.subheader("📂 Upload Transaction Dataset")

uploaded_file = st.file_uploader(
    "Choose CSV File",
    type=["csv"]
)

# ==================================================
# PROCESS FILE
# ==================================================

if uploaded_file is not None:

    try:

        df = pd.read_csv(uploaded_file)

        st.success("Dataset Loaded Successfully")

        col1, col2 = st.columns([2,1])

        with col1:
            st.subheader("Dataset Preview")
            st.dataframe(
                df.head(),
                use_container_width=True
            )

        with col2:
            st.subheader("Dataset Information")

            st.metric(
                "Rows",
                df.shape[0]
            )

            st.metric(
                "Columns",
                df.shape[1]
            )

        st.markdown("---")

        analyze = st.button(
            "🚀 Run Fraud Analysis",
            use_container_width=True
        )

        if analyze:

            # ==========================================
            # KEEP YOUR EXISTING LOGIC BELOW
            # ==========================================

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
                len(scaled_features)-SEQ_LEN
            ):
                sequences.append(
                    scaled_features[
                        i:i+SEQ_LEN
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

            # ==========================================
            # KPI DASHBOARD
            # ==========================================

            st.markdown("## 📊 Executive Summary")

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "Transactions",
                    len(results)
                )

            with c2:
                st.metric(
                    "Avg Fraud Risk",
                    f"{avg_prob:.2f}%"
                )

            with c3:
                st.metric(
                    "High Risk Cases",
                    len(high_risk)
                )

            st.progress(
                min(avg_prob/100,1.0)
            )

            # ==========================================
            # TABS
            # ==========================================

            tab1, tab2, tab3 = st.tabs(
                [
                    "🚨 Detection Results",
                    "📈 Analytics",
                    "⬇️ Export"
                ]
            )

            # ------------------------------------------

            with tab1:

                st.subheader(
                    "High Risk Transactions"
                )

                if len(high_risk):

                    st.error(
                        f"{len(high_risk)} suspicious transactions detected"
                    )

                    st.dataframe(
                        high_risk,
                        use_container_width=True
                    )

                else:

                    st.success(
                        "No suspicious transactions found."
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

            # ------------------------------------------

            with tab2:

                st.subheader(
                    "Fraud Probability Distribution"
                )

                fig = px.histogram(
                    results,
                    x="Fraud_Probability",
                    nbins=30,
                    title="Fraud Risk Distribution"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
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

            # ------------------------------------------

            with tab3:

                csv = results.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="📥 Download Full Report",
                    data=csv,
                    file_name="fraud_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    except Exception as e:

        st.error(
            f"Prediction Error: {str(e)}"
        )