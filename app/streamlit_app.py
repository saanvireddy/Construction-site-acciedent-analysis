import sys
import os
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, 'src')

import streamlit as st
import pandas as pd
import joblib
import json
from predict import predict_risk, load_model_and_vectorizer
from ner import get_top_equipment, get_top_hazards

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Construction Accident Risk Analyzer",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .high-risk-box {
        background: linear-gradient(135deg, #ff4444, #cc0000);
        border-radius: 12px; padding: 20px;
        color: white; text-align: center;
        box-shadow: 0 4px 15px rgba(255,68,68,0.4);
    }
    .low-risk-box {
        background: linear-gradient(135deg, #00c851, #007E33);
        border-radius: 12px; padding: 20px;
        color: white; text-align: center;
        box-shadow: 0 4px 15px rgba(0,200,81,0.4);
    }
    .metric-card {
        background: #1e2130; border-radius: 10px;
        padding: 15px; text-align: center;
        border: 1px solid #2d3250;
    }
    .stTextArea textarea {
        font-size: 14px !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    return load_model_and_vectorizer()


@st.cache_data
def load_data():
    if os.path.exists('data/processed.csv'):
        return pd.read_csv('data/processed.csv')
    return None


@st.cache_data
def load_results():
    if os.path.exists('models/training_results.json'):
        with open('models/training_results.json') as f:
            return json.load(f)
    return None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/hard-hat.png", width=80)
    st.title("⚠️ Accident Risk\nAnalyzer")
    st.markdown("---")

    st.markdown("### 📊 Model Info")
    results = load_results()
    if results and 'best_model_metrics' in results:
        m = results['best_model_metrics']
        st.metric("Best Model", m.get('model_name', 'SVM'))
        st.metric("AUC-ROC",   f"{m.get('auc_roc', 0):.4f}")
        st.metric("F1 Score",  f"{m.get('f1', 0):.4f}")
        st.metric("Precision", f"{m.get('precision', 0):.4f}")
        st.metric("Recall",    f"{m.get('recall', 0):.4f}")

    st.markdown("---")
    st.markdown("### 📁 Dataset")
    df = load_data()
    if df is not None:
        st.metric("Total Records",       f"{len(df):,}")
        st.metric("Construction Reports", f"{len(df):,}")
        high = df['high_risk'].sum() if 'high_risk' in df.columns else 0
        st.metric("High Risk Reports",   f"{high:,}")

    st.markdown("---")
    st.markdown("**Built with:** Python · scikit-learn · spaCy · NLTK · Streamlit · Docker")
    st.markdown("**Data:** OSHA Severe Injury Reports (2015–2025)")


# ── Main content ──────────────────────────────────────────────────────────────
st.title("🏗️ Construction Site Accident Risk Analyzer")
st.markdown(
    "Analyze accident reports using **NLP + Machine Learning** to classify "
    "risk levels and generate **AI-powered safety recommendations**."
)
st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔍 Analyze Report",
    "📊 Model Performance",
    "📈 Data Insights"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ANALYZE REPORT
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("📋 Enter Accident Report")

    # Sample reports
    sample_reports = {
        "Select a sample...": "",
        "Fall from Scaffold": """Worker fell from a scaffold while using a nail gun 
to install roof panels. The scaffold collapsed causing the worker to fall 
15 feet and hit the concrete floor. Worker was hospitalized with broken 
collarbone and head injury.""",
        "Electrocution": """Electrician was electrocuted while working on an 
unguarded power panel. Worker contacted a live wire and was thrown backwards. 
Rushed to hospital with severe burns on both hands.""",
        "Equipment Caught-In": """Worker's hand was caught in a concrete mixer 
while attempting to clear a blockage without shutting down the machine. 
Worker sustained amputation of two fingers on right hand.""",
        "Minor Slip": """Worker slipped on wet floor near the break area 
and bruised their knee. First aid was administered on site. 
Worker returned to work the same day without medical attention.""",
    }

    selected = st.selectbox("Or choose a sample report:", list(sample_reports.keys()))
    default_text = sample_reports[selected]

    report_text = st.text_area(
        "Paste accident report narrative here:",
        value=default_text,
        height=180,
        placeholder="e.g. Worker fell from scaffolding while installing roof panels..."
    )

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)
    with col_btn2:
        clear_btn = st.button("🗑️ Clear", use_container_width=False)

    if analyze_btn and report_text.strip():
        model, vectorizer = load_models()

        with st.spinner("Analyzing accident report..."):
            result = predict_risk(
                report_text,
                model=model,
                vectorizer=vectorizer,
                include_recommendation=True
            )

        st.markdown("---")

        # Risk level display
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            if result['prediction'] == 1:
                st.markdown(f"""
                <div class="high-risk-box">
                    <h2>🔴 HIGH RISK</h2>
                    <h1>{result['probability']:.1%}</h1>
                    <p>Probability of High-Risk Injury</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="low-risk-box">
                    <h2>🟢 LOW RISK</h2>
                    <h1>{result['probability']:.1%}</h1>
                    <p>Probability of High-Risk Injury</p>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown("#### 🏷️ Equipment Detected")
            if result['entities']['equipment']:
                for eq in result['entities']['equipment']:
                    st.markdown(f"• `{eq}`")
            else:
                st.markdown("_None detected_")

            st.markdown("#### ⚡ Hazard Types")
            if result['entities']['hazards']:
                for hz in result['entities']['hazards']:
                    st.markdown(f"• `{hz}`")
            else:
                st.markdown("_None detected_")

        with col3:
            st.markdown("#### 🔑 Key Risk Factors")
            for feat in result['top_features'][:8]:
                st.markdown(f"• `{feat}`")

        # Safety recommendation
        if result['recommendation']:
            st.markdown("---")
            st.subheader("🛡️ Safety Recommendation")
            st.markdown(result['recommendation'])

    elif analyze_btn:
        st.warning("⚠️ Please enter an accident report to analyze.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 Model Training Results")

    results = load_results()
    if results:
        # Model comparison table
        models_data = {
            k: v for k, v in results.items()
            if k != 'best_model_metrics' and 'cv_f1_mean' in v
        }

        if models_data:
            df_models = pd.DataFrame([
                {
                    'Model':     name,
                    'CV F1 Mean': f"{v['cv_f1_mean']:.4f}",
                    'CV F1 Std':  f"± {v['cv_f1_std']:.4f}",
                    'Status':    '🏆 Best' if name == results.get(
                        'best_model_metrics', {}).get('model_name') else ''
                }
                for name, v in models_data.items()
            ])
            st.dataframe(df_models, use_container_width=True, hide_index=True)

    # Show generated charts
    col1, col2 = st.columns(2)

    with col1:
        if os.path.exists('reports/model_comparison.png'):
            st.image('reports/model_comparison.png',
                     caption='Model Comparison — CV F1 Scores',
                     )

    with col2:
        if os.path.exists('reports/roc_curve.png'):
            st.image('reports/roc_curve.png',
                     caption='ROC Curve — Best Model',
                    )

    if os.path.exists('reports/confusion_matrix.png'):
        st.image('reports/confusion_matrix.png',
                 caption='Confusion Matrix',
                 )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DATA INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📈 OSHA Construction Accident Insights (2015–2025)")

    col1, col2 = st.columns(2)

    with col1:
        if os.path.exists('reports/accidents_over_time.png'):
            st.image('reports/accidents_over_time.png',
                     caption='Accidents Over Time',
                     )

        if os.path.exists('reports/wordcloud_high_risk.png'):
            st.image('reports/wordcloud_high_risk.png',
                     caption='Most Common Words in High-Risk Reports',
                     )

    with col2:
        if os.path.exists('reports/risk_by_state.png'):
            st.image('reports/risk_by_state.png',
                     caption='High-Risk Accidents by State',
                     )

        # Top equipment and hazards
        df = load_data()
        if df is not None:
            st.markdown("#### 🔧 Most Common Equipment in Accidents")
            top_eq = get_top_equipment(df, text_col='Final Narrative')
            if top_eq:
                eq_df = pd.DataFrame(top_eq, columns=['Equipment', 'Count'])
                st.dataframe(eq_df, use_container_width=True, hide_index=True)

            st.markdown("#### ⚡ Most Common Hazard Types")
            top_hz = get_top_hazards(df, text_col='Final Narrative')
            if top_hz:
                hz_df = pd.DataFrame(top_hz, columns=['Hazard', 'Count'])
                st.dataframe(hz_df, use_container_width=True, hide_index=True)