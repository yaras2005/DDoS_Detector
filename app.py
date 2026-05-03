import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st

from src.inference import analyze_uploaded_file

st.set_page_config(
    page_title="NetShield AI — DDoS Detection Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    background-color: #0a0e1a;
    color: #e2e8f0;
}
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1526 50%, #0a0e1a 100%);
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1526 0%, #111827 100%);
    border-right: 1px solid #1e3a5f;
}
section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
.top-header {
    background: linear-gradient(90deg, #0d1f3c 0%, #1a3a6b 50%, #0d1f3c 100%);
    border-bottom: 2px solid #2e75b6;
    padding: 20px 32px;
    margin: -1rem -1rem 2rem -1rem;
    display: flex;
    align-items: center;
    gap: 16px;
}
.top-header h1 { font-size: 1.8rem; font-weight: 700; color: #ffffff !important; margin: 0; letter-spacing: -0.5px; }
.top-header span { color: #60a5fa; }
.header-badge {
    background: #1e3a5f; border: 1px solid #2e75b6; color: #60a5fa !important;
    padding: 4px 12px; border-radius: 20px; font-size: 0.75rem;
    font-weight: 600; letter-spacing: 1px; text-transform: uppercase;
}
.section-title {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 2px;
    text-transform: uppercase; color: #2e75b6 !important;
    margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid #1e3a5f;
}
.card {
    background: linear-gradient(135deg, #0d1f3c 0%, #111827 100%);
    border: 1px solid #1e3a5f; border-radius: 12px;
    padding: 20px 24px; margin-bottom: 16px;
}
.metric-card {
    background: linear-gradient(135deg, #0d1f3c 0%, #0f2645 100%);
    border: 1px solid #1e3a5f; border-radius: 12px;
    padding: 20px; text-align: center; transition: border-color 0.2s;
}
.metric-card:hover { border-color: #2e75b6; }
.metric-label { font-size: 0.7rem; font-weight: 600; letter-spacing: 1.5px; text-transform: uppercase; color: #64748b; margin-bottom: 8px; }
.metric-value { font-size: 2rem; font-weight: 700; color: #60a5fa; line-height: 1; }
.metric-value.danger { color: #f87171; }
.metric-value.success { color: #34d399; }
.metric-value.warning { color: #fbbf24; }
.alert-danger {
    background: linear-gradient(90deg, #1f0a0a, #2d1010);
    border: 1px solid #dc2626; border-left: 4px solid #dc2626;
    border-radius: 8px; padding: 16px 20px;
    color: #fca5a5 !important; font-weight: 600; font-size: 1rem; margin-bottom: 20px;
}
.alert-success {
    background: linear-gradient(90deg, #0a1f0f, #0d2b14);
    border: 1px solid #16a34a; border-left: 4px solid #16a34a;
    border-radius: 8px; padding: 16px 20px;
    color: #86efac !important; font-weight: 600; font-size: 1rem; margin-bottom: 20px;
}
.badge {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase;
}
.badge-danger { background: #450a0a; color: #f87171; border: 1px solid #dc2626; }
.badge-success { background: #052e16; color: #34d399; border: 1px solid #16a34a; }
.badge-warning { background: #431407; color: #fbbf24; border: 1px solid #d97706; }
.badge-info { background: #0c1a2e; color: #60a5fa; border: 1px solid #2563eb; }
.badge-purple { background: #1e1040; color: #a78bfa; border: 1px solid #7c3aed; }
.stDataFrame { border: 1px solid #1e3a5f !important; border-radius: 8px !important; }
.stButton > button {
    background: linear-gradient(90deg, #1a3a6b, #2e75b6) !important;
    color: white !important; border: none !important; border-radius: 8px !important;
    padding: 10px 32px !important; font-weight: 600 !important;
    font-size: 0.9rem !important; letter-spacing: 0.5px !important; transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #2e75b6, #3b82f6) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(46, 117, 182, 0.4) !important;
}
.stSpinner > div { border-top-color: #2e75b6 !important; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
.divider { border: none; border-top: 1px solid #1e3a5f; margin: 24px 0; }
.info-row { display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #1e3a5f; font-size: 0.88rem; }
.info-label { color: #64748b; font-weight: 500; }
.info-value { color: #e2e8f0; font-weight: 600; }
.hint-card {
    background: #0d1f3c; border: 1px solid #7c3aed;
    border-left: 4px solid #7c3aed; border-radius: 8px;
    padding: 14px 18px; margin-bottom: 10px;
}
.hint-title { color: #a78bfa; font-weight: 700; font-size: 0.82rem; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 8px; }
.hint-signal { color: #cbd5e1; font-size: 0.82rem; padding: 3px 0; }
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="top-header">
    <div style="font-size:2rem">🛡️</div>
    <div>
        <h1>NetShield <span>AI</span></h1>
        <div style="color:#64748b; font-size:0.8rem; margin-top:2px;">
            Hybrid DDoS Detection Platform — Lebanese American University
        </div>
    </div>
    <div style="margin-left:auto; display:flex; gap:8px; align-items:center;">
        <span class="header-badge">ML + Rule-Based</span>
        <span class="header-badge">Multi-Class</span>
        <span class="header-badge">Real-Time</span>
    </div>
</div>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown('<div class="section-title">System Status</div>', unsafe_allow_html=True)

    model_ok = os.path.exists("models/best_model.pkl")
    schema_ok = os.path.exists("artifacts/preprocessor_columns.json")
    encoder_ok = os.path.exists("artifacts/label_encoder.pkl")

    st.markdown(f"""
    <div style="display:flex; flex-direction:column; gap:8px; margin-bottom:20px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="font-size:0.85rem;">ML Model</span>
            <span class="badge {'badge-success' if model_ok else 'badge-danger'}">
                {'LOADED' if model_ok else 'MISSING'}
            </span>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="font-size:0.85rem;">Feature Schema</span>
            <span class="badge {'badge-success' if schema_ok else 'badge-danger'}">
                {'LOADED' if schema_ok else 'MISSING'}
            </span>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="font-size:0.85rem;">Label Encoder</span>
            <span class="badge {'badge-success' if encoder_ok else 'badge-danger'}">
                {'LOADED' if encoder_ok else 'MISSING'}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Detection Engine</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.82rem; color:#94a3b8; line-height:1.6; margin-bottom:16px;">
        This system uses a <strong style="color:#60a5fa;">two-layer hybrid approach</strong>:
        a lightweight rule-based anomaly filter followed by a trained
        machine learning classifier for final prediction.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Processing Settings</div>', unsafe_allow_html=True)
    max_rows = st.slider("Max rows to process", min_value=1000, max_value=100000, value=50000, step=1000)

    attack_threshold = st.slider(
        "Attack ratio threshold",
        min_value=0.05,
        max_value=0.80,
        value=0.30,
        step=0.05,
        help="If the proportion of predicted attack rows exceeds this, the system flags the traffic."
    )

    st.markdown('<div class="section-title">About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.78rem; color:#64748b; line-height:1.7;">
        <div>📚 CSC 437 — Network Security</div>
        <div>🏛️ Lebanese American University</div>
        <div>👥 Joseph Al Khoury</div>
        <div style="padding-left:22px;">Yara Saade</div>
        <div style="padding-left:22px;">Juliano Bou Abdallah</div>
        <div>🗓️ Spring 2026</div>
    </div>
    """, unsafe_allow_html=True)


st.markdown('<div class="section-title">Traffic Input</div>', unsafe_allow_html=True)

col_upload, col_label = st.columns([3, 1])
with col_upload:
    uploaded_file = st.file_uploader(
        "Upload a network traffic CSV file", type=["csv"],
        help="Upload a CSV containing network flow features"
    )
with col_label:
    label_column = st.text_input(
        "Label column to exclude", value="Attack Type",
        help="This column will be dropped before prediction"
    )

if uploaded_file is None:
    st.markdown("""
    <div class="card" style="text-align:center; padding:48px;">
        <div style="font-size:3rem; margin-bottom:16px;">📡</div>
        <div style="font-size:1.1rem; font-weight:600; color:#e2e8f0; margin-bottom:8px;">
            No traffic data loaded
        </div>
        <div style="font-size:0.85rem; color:#64748b;">
            Upload a CSV file above to begin real-time DDoS analysis
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


file_size_mb = uploaded_file.size / (1024 * 1024)

try:
    df = pd.read_csv(uploaded_file, nrows=max_rows)
except Exception as e:
    st.error(f"Error reading CSV file: {e}")
    st.stop()

st.markdown(f"""
<div class="card" style="padding:14px 20px;">
    <div style="display:flex; gap:32px; align-items:center; flex-wrap:wrap;">
        <div><span class="info-label">File: </span><span class="info-value">{uploaded_file.name}</span></div>
        <div><span class="info-label">Size: </span><span class="info-value">{file_size_mb:.2f} MB</span></div>
        <div><span class="info-label">Rows loaded: </span><span class="info-value">{len(df):,}</span></div>
        <div><span class="info-label">Features: </span><span class="info-value">{len(df.columns)}</span></div>
        <div style="margin-left:auto;"><span class="badge badge-info">READY FOR ANALYSIS</span></div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-title" style="margin-top:24px;">Data Preview</div>', unsafe_allow_html=True)
st.dataframe(df.head(10), use_container_width=True)
st.markdown("<hr class='divider'>", unsafe_allow_html=True)


col_btn, col_note = st.columns([1, 4])
with col_btn:
    analyze = st.button("🔍 Analyze Traffic", type="primary", use_container_width=True)
with col_note:
    st.markdown(
        "<div style='padding-top:10px; color:#64748b; font-size:0.82rem;'>"
        "Runs rule-based screening followed by ML classification on all loaded rows."
        "</div>", unsafe_allow_html=True
    )

if not analyze:
    st.stop()


if not model_ok:
    st.error("❌ ML model not found. Run `test_pipeline.py` first.")
    st.stop()
if not schema_ok:
    st.error("❌ Feature schema not found. Run `test_pipeline.py` first.")
    st.stop()
if not encoder_ok:
    st.error("❌ Label encoder not found. Run `test_pipeline.py` first.")
    st.stop()


with st.spinner("Running hybrid detection engine..."):
   
    result = analyze_uploaded_file(df, label_column=label_column or None, attack_threshold=attack_threshold)


st.markdown('<div class="section-title" style="margin-top:8px;">Detection Result</div>', unsafe_allow_html=True)

if result["final_flag"]:
    st.markdown(f'<div class="alert-danger">⚠️ &nbsp; {result["final_label"]}</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="alert-success">✅ &nbsp; {result["final_label"]}</div>', unsafe_allow_html=True)


m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f"""
    <div class="metric-card"><div class="metric-label">Confidence</div>
    <div class="metric-value {'danger' if result['confidence'] > 50 else 'success'}">{result['confidence']}%</div></div>
    """, unsafe_allow_html=True)
with m2:
    st.markdown(f"""
    <div class="metric-card"><div class="metric-label">Attack Rows</div>
    <div class="metric-value danger">{result['attack_count']:,}</div></div>
    """, unsafe_allow_html=True)
with m3:
    st.markdown(f"""
    <div class="metric-card"><div class="metric-label">Normal Rows</div>
    <div class="metric-value success">{result['normal_count']:,}</div></div>
    """, unsafe_allow_html=True)
with m4:
    st.markdown(f"""
    <div class="metric-card"><div class="metric-label">Attack Ratio</div>
    <div class="metric-value {'danger' if result['attack_ratio'] > 30 else 'warning'}">{result['attack_ratio']}%</div></div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


if "class_counts" in result and result["class_counts"]:
    st.markdown('<div class="section-title">Traffic Classification Breakdown</div>', unsafe_allow_html=True)

    class_df = pd.DataFrame(
        list(result["class_counts"].items()), columns=["Class", "Count"]
    ).sort_values("Count", ascending=False).reset_index(drop=True)
    class_df["Percentage"] = (class_df["Count"] / result["total_rows"] * 100).round(2)

    col_chart, col_pie = st.columns(2)
    with col_chart:
        fig_bar = px.bar(
            class_df, x="Class", y="Count", color="Count",
            color_continuous_scale=["#1e3a5f", "#2e75b6", "#60a5fa"], text="Count",
        )
        fig_bar.update_traces(textposition="outside", textfont_color="#e2e8f0")
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,31,60,0.5)",
            font_color="#e2e8f0", coloraxis_showscale=False,
            xaxis=dict(gridcolor="#1e3a5f", title=""),
            yaxis=dict(gridcolor="#1e3a5f", title="Row Count"),
            margin=dict(t=20, b=20), height=320,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_pie:
        colors = ["#34d399", "#f87171", "#fbbf24", "#60a5fa", "#a78bfa", "#fb923c", "#f472b6"]
        fig_pie = px.pie(
            class_df, names="Class", values="Count",
            color_discrete_sequence=colors, hole=0.45,
        )
        fig_pie.update_traces(textposition="outside", textinfo="percent+label", textfont_color="#e2e8f0")
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
            showlegend=False, margin=dict(t=20, b=20), height=320,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown('<div class="section-title" style="margin-top:8px;">Per-Class Metrics</div>', unsafe_allow_html=True)

    if os.path.exists("results/per_class_metrics.json"):
        with open("results/per_class_metrics.json", "r") as f:
            per_class = json.load(f)
        per_class_df = pd.DataFrame(per_class).T.reset_index()
        per_class_df.columns = ["Class", "Precision", "Recall", "F1-Score", "Support"]
        per_class_df = per_class_df[per_class_df["Class"].str.lower() != "accuracy"]
        for col in ["Precision", "Recall", "F1-Score"]:
            per_class_df[col] = per_class_df[col].apply(lambda x: f"{float(x):.4f}")
        st.dataframe(per_class_df, use_container_width=True, hide_index=True)
    else:
        st.info("Per-class metrics not found. Run `test_pipeline.py` to generate them.")

    class_df["Percentage"] = class_df["Percentage"].astype(str) + "%"
    st.dataframe(class_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title" style="margin-top:8px;">Export Results</div>', unsafe_allow_html=True)
    if "predictions_labels" in result:
        export_df = df.head(len(result["predictions_labels"])).copy().reset_index(drop=True)
        export_df["Predicted Class"] = result["predictions_labels"]
        export_df["Rule Flagged"] = result.get("rule_flag", False)
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Full Predictions as CSV",
            data=csv_bytes,
            file_name="netshield_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )


st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown('<div class="section-title">Detection Engine Details</div>', unsafe_allow_html=True)

col_ml, col_rule = st.columns(2)

with col_ml:
    st.markdown(f"""
    <div class="card">
        <div style="font-size:0.7rem; letter-spacing:1.5px; text-transform:uppercase;
                    color:#2e75b6; margin-bottom:12px; font-weight:700;">ML Classification Layer</div>
        <div class="info-row">
            <span class="info-label">Deployed Model</span>
            <span class="info-value">{result['best_model'].replace('_', ' ').title()}</span>
        </div>
        <div class="info-row">
            <span class="info-label">ML Decision</span>
            <span class="badge {'badge-danger' if result['ml_flag'] else 'badge-success'}">
                {'Suspicious' if result['ml_flag'] else 'Normal'}
            </span>
        </div>
        <div class="info-row" style="border:none;">
            <span class="info-label">Confidence Score</span>
            <span class="info-value">{result['confidence']}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_rule:
    st.markdown(f"""
    <div class="card">
        <div style="font-size:0.7rem; letter-spacing:1.5px; text-transform:uppercase;
                    color:#2e75b6; margin-bottom:12px; font-weight:700;">Rule-Based Screening Layer</div>
        <div class="info-row">
            <span class="info-label">Rule Engine</span>
            <span class="badge {'badge-danger' if result['rule_flag'] else 'badge-success'}">
                {'Triggered' if result['rule_flag'] else 'Not Triggered'}
            </span>
        </div>
        <div class="info-row">
            <span class="info-label">Anomaly Score</span>
            <span class="info-value">{result['rule_score']}</span>
        </div>
        <div class="info-row" style="border:none;">
            <span class="info-label">Reason</span>
            <span class="info-value" style="font-size:0.78rem; max-width:60%; text-align:right;">
                {result['rule_reason']}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

if result.get("rule_attack_hints"):
    st.markdown('<div class="section-title" style="margin-top:4px;">Rule-Based Attack Type Signals</div>', unsafe_allow_html=True)
    hint_cols = st.columns(min(len(result["rule_attack_hints"]), 3))
    for i, hint in enumerate(result["rule_attack_hints"]):
        with hint_cols[i % 3]:
            signals_html = "".join([
                f'<div class="hint-signal">• {s}</div>'
                for s in hint["signals"]
            ])
            st.markdown(f"""
            <div class="hint-card">
                <div class="hint-title">⚠ {hint['attack_type']}</div>
                {signals_html}
            </div>
            """, unsafe_allow_html=True)

if result["suspicious_columns"]:
    st.markdown(f"""
    <div class="card" style="margin-top:0;">
        <div style="font-size:0.7rem; letter-spacing:1.5px; text-transform:uppercase;
                    color:#fbbf24; margin-bottom:10px; font-weight:700;">
            ⚡ Suspicious Feature Columns Detected
        </div>
        <div style="display:flex; flex-wrap:wrap; gap:8px;">
            {''.join([f'<span class="badge badge-warning">{col}</span>' for col in result["suspicious_columns"]])}
        </div>
    </div>
    """, unsafe_allow_html=True)


st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown('<div class="section-title">Professional Interpretation</div>', unsafe_allow_html=True)

if result["final_flag"]:
    interpretation = (
        "This traffic sample should be investigated because the hybrid detector found signs "
        "that match attack-like behavior. In a real environment, the next step would be "
        "rate limiting, alerting, and deeper packet inspection."
    )
    icon, color = "🚨", "#fca5a5"
else:
    interpretation = (
        "This traffic sample does not show enough evidence of DDoS activity under the current "
        "model and screening rules. In a real deployment, the sample would still be logged "
        "for auditing purposes."
    )
    icon, color = "✅", "#86efac"

st.markdown(f"""
<div class="card">
    <div style="display:flex; gap:16px; align-items:flex-start;">
        <div style="font-size:1.8rem;">{icon}</div>
        <div style="font-size:0.9rem; line-height:1.7; color:{color};">{interpretation}</div>
    </div>
</div>
""", unsafe_allow_html=True)


st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown('<div class="section-title">Model Training Performance</div>', unsafe_allow_html=True)

if os.path.exists("results/model_comparison.csv"):
    results_df = pd.read_csv("results/model_comparison.csv")
    display_cols = [c for c in ["Model", "Accuracy", "Precision", "Recall", "F1-score"] if c in results_df.columns]
    st.dataframe(results_df[display_cols], use_container_width=True, hide_index=True)

col_img1, col_img2 = st.columns(2)
with col_img1:
    if os.path.exists("results/accuracy_comparison.png"):
        st.image("results/accuracy_comparison.png", caption="Model Accuracy Comparison", use_container_width=True)
with col_img2:
    if os.path.exists("results/best_model.json"):
        with open("results/best_model.json", "r", encoding="utf-8") as f:
            best_meta = json.load(f)
        best_model_name = best_meta.get("best_model", "")
        conf_path = f"results/{best_model_name}_confusion_matrix.png"
        if os.path.exists(conf_path):
            st.image(conf_path, caption=f"Confusion Matrix — {best_model_name}", use_container_width=True)


st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; padding:16px 0; color:#334155; font-size:0.78rem;">
    NetShield AI &nbsp;·&nbsp; CSC 437 Network Security &nbsp;·&nbsp;
    Lebanese American University &nbsp;·&nbsp; Spring 2026
</div>
""", unsafe_allow_html=True)