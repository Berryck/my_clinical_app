import streamlit as st
import streamlit.components.v1 as components
import pickle
import joblib
import numpy as np
import shap
import pandas as pd
import matplotlib.pyplot as plt

# ========== IMPORTANT: set_page_config must be the first st command ==========
st.set_page_config(page_title="Clinical Decision Support System", layout="wide", page_icon="🏥")

# --- 0. Helper functions ---
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height if height else 150, scrolling=True)

# --- 1. Cache resource loading ---
@st.cache_resource
def load_artifacts():
    model = joblib.load("saved_models/LightGBM_Optimized.pkl")

    try:
        scaler = joblib.load("saved_models/scaler1.pkl")
    except Exception:
        with open("saved_models/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

    try:
        feature_names = joblib.load("saved_models/feature_names1.pkl")
    except Exception:
        with open("saved_models/feature_names.pkl", "rb") as f:
            feature_names = pickle.load(f)

    return model, scaler, feature_names

# Initialize loading (after set_page_config)
try:
    model, scaler, feature_names = load_artifacts()
except FileNotFoundError:
    st.error("❌ Error: Model files not found. Please check the saved_models/ directory for .pkl files.")
    st.stop()

# --- 2. Page title ---
st.title("🏥 LightGBM-based Clinical Risk Prediction System")

# --- 3. Create tabs ---
tab1, tab2 = st.tabs(["📝 Single Prediction (Manual Input)", "📂 Batch Prediction (Upload Excel)"])

# ==========================================
# Mode 1: Single Prediction
# ==========================================
with tab1:
    st.info("Suitable for rapid risk assessment and attribution analysis for a single patient.")

    with st.form("single_predict_form"):
        inputs = {}
        n_cols = 4 if len(feature_names) > 10 else 2
        cols = st.columns(n_cols)

        for i, feat in enumerate(feature_names):
            with cols[i % n_cols]:
                inputs[feat] = st.number_input(f"{feat}", value=0.0, format="%.4f")

        submitted = st.form_submit_button("🚀 Start Prediction")

    if submitted:
        x_df = pd.DataFrame([inputs], columns=feature_names)

        try:
            x_scaled = scaler.transform(x_df)

            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(x_scaled)[0, 1]
            else:
                prob = model.predict(x_scaled)[0]

            st.divider()
            c1, c2 = st.columns([1, 2])
            with c1:
                st.subheader("Prediction Result")
                st.metric("Risk Probability", f"{prob * 100:.2f}%")
                if prob > 0.5:
                    st.error("🔴 High Risk")
                else:
                    st.success("🟢 Low Risk")

            with c2:
                st.subheader("Individual Attribution Analysis")
                with st.spinner("Calculating feature contributions..."):
                    explainer = shap.TreeExplainer(model)
                    shap_values_all = explainer.shap_values(x_scaled)

                    if isinstance(shap_values_all, list):
                        shap_values = shap_values_all[1]
                        base_value = explainer.expected_value[1]
                    else:
                        shap_values = shap_values_all
                        base_value = explainer.expected_value
                        if isinstance(base_value, np.ndarray):
                            base_value = base_value[0]

                    st.markdown("**1. Waterfall Plot**")
                    explanation = shap.Explanation(
                        values=shap_values[0],
                        base_values=base_value,
                        data=x_df.iloc[0],
                        feature_names=feature_names
                    )
                    fig = plt.figure(figsize=(10, 5))
                    shap.plots.waterfall(explanation, max_display=10, show=False)
                    st.pyplot(fig, bbox_inches='tight')
                    plt.close(fig)

                    st.markdown("**2. Force Plot**")
                    st.caption("Hover over the plot to see specific values.")
                    force_plot_html = shap.force_plot(
                        base_value,
                        shap_values[0],
                        x_df.iloc[0],
                        feature_names=feature_names,
                        matplotlib=False
                    )
                    st_shap(force_plot_html, height=160)

        except Exception as e:
            st.error(f"Error during execution: {e}")
            import traceback
            st.text(traceback.format_exc())

# ==========================================
# Mode 2: Batch Prediction
# ==========================================
with tab2:
    st.info("Suitable for processing multiple records. Please upload an Excel (.xlsx) or CSV file.")

    with st.expander("📥 Download Data Template"):
        st.write("Please ensure your file contains the following columns:")
        st.code(str(feature_names), language="python")
        template_df = pd.DataFrame(columns=['Patient_ID'] + feature_names)
        csv = template_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV Template", csv, "prediction_template.csv", "text/csv")

    uploaded_file = st.file_uploader("Upload file", type=["xlsx", "csv"])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                try:
                    df_upload = pd.read_csv(uploaded_file)
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    df_upload = pd.read_csv(uploaded_file, encoding='gbk')
            else:
                df_upload = pd.read_excel(uploaded_file)

            st.write(f"✅ Successfully read {len(df_upload)} records.")

            df_upload.columns = df_upload.columns.str.strip()
            missing_cols = [col for col in feature_names if col not in df_upload.columns]

            if missing_cols:
                st.error(f"❌ File is missing the following required feature columns:\n{missing_cols}")
            else:
                X_batch = df_upload[feature_names]
                X_batch_scaled = scaler.transform(X_batch)

                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_batch_scaled)[:, 1]
                else:
                    probs = model.predict(X_batch_scaled)

                df_result = df_upload.copy()
                df_result['Predicted Probability'] = np.round(probs, 4)
                df_result['Risk Level'] = [
                    'High Risk' if p > 0.5 else 'Low Risk' for p in probs
                ]

                st.subheader("📊 Prediction Results Overview")
                st.dataframe(df_result.style.map(
                    lambda x: 'background-color: #ffcccc'
                    if x == 'High Risk' else 'background-color: #ccffcc',
                    subset=['Risk Level']
                ))

                csv_result = df_result.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    "💾 Download Prediction Results (.csv)",
                    csv_result,
                    "prediction_results.csv",
                    "text/csv"
                )

                st.divider()
                st.subheader("🔍 In-depth Analysis: View SHAP Explanation for a Specific Patient")
                selected_index = st.selectbox(
                    "Select the row index to analyze",
                    options=df_result.index,
                    format_func=lambda
                        x: f"Row {x} (Probability: {df_result.loc[x, 'Predicted Probability']:.2%})"
                )

                if st.button("Explain this patient"):
                    x_single_df = X_batch.iloc[[selected_index]]
                    x_single_scaled = X_batch_scaled[selected_index].reshape(1, -1)

                    explainer = shap.TreeExplainer(model)
                    shap_values_all = explainer.shap_values(x_single_scaled)

                    if isinstance(shap_values_all, list):
                        sv = shap_values_all[1][0]
                        bv = explainer.expected_value[1]
                    else:
                        sv = shap_values_all[0]
                        bv = explainer.expected_value
                        if isinstance(bv, np.ndarray):
                            bv = bv[0]

                    st.markdown("**1. Waterfall Plot**")
                    exp = shap.Explanation(
                        values=sv, base_values=bv,
                        data=x_single_df.iloc[0],
                        feature_names=feature_names
                    )
                    fig_batch = plt.figure(figsize=(10, 5))
                    shap.plots.waterfall(exp, max_display=10, show=False)
                    st.pyplot(fig_batch, bbox_inches='tight')
                    plt.close(fig_batch)

                    st.markdown("**2. Force Plot**")
                    force_plot_html_batch = shap.force_plot(
                        bv,
                        sv,
                        x_single_df.iloc[0],
                        feature_names=feature_names,
                        matplotlib=False
                    )
                    st_shap(force_plot_html_batch, height=160)

        except Exception as e:
            st.error(f"Error processing file: {e}")
