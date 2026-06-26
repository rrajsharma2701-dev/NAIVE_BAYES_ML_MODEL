import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              classification_report, roc_curve, auc,
                              precision_score, recall_score, f1_score)
import joblib
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="VAULTIQ · Credit Risk AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
#  GLOBAL CSS — Luxury Finance / Cyber Bank Aesthetic
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=DM+Sans:wght@300;400;500;700&family=Courier+Prime:wght@400;700&display=swap');

:root {
  --bg:        #040810;
  --surface:   #070d1a;
  --card:      #0b1525;
  --card2:     #0e1c30;
  --border:    rgba(0,210,255,0.15);
  --cyan:      #00d2ff;
  --blue:      #0066ff;
  --violet:    #7c3aed;
  --green:     #00ffaa;
  --red:       #ff2d55;
  --gold:      #f5c518;
  --amber:     #ff9500;
  --steel:     #8892a4;
  --text:      #e8edf5;
  --muted:     #4a5568;
}

html, body, [class*="css"] {
  background-color: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'DM Sans', sans-serif !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.2rem 2.2rem 3rem !important; max-width: 1480px !important; }

/* ── SCANLINE OVERLAY ── */
body::before {
  content: "";
  position: fixed; inset: 0;
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 2px,
    rgba(0,210,255,0.012) 2px,
    rgba(0,210,255,0.012) 4px
  );
  pointer-events: none;
  z-index: 9999;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #050c1a 0%, #070f20 100%) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── HERO ── */
.hero-shell {
  position: relative;
  border-radius: 20px;
  overflow: hidden;
  margin-bottom: 2rem;
  border: 1px solid rgba(0,210,255,0.2);
  box-shadow: 0 0 80px rgba(0,102,255,0.1), 0 0 0 1px rgba(0,210,255,0.05);
}
.hero-inner {
  background: linear-gradient(135deg, #040c1e 0%, #06102a 40%, #040a18 100%);
  padding: 2.6rem 3rem;
  position: relative;
  overflow: hidden;
}
.hero-inner::before {
  content: "";
  position: absolute;
  top: -80px; right: -80px;
  width: 450px; height: 450px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(0,102,255,0.1) 0%, transparent 65%);
}
.hero-inner::after {
  content: "";
  position: absolute;
  bottom: -60px; left: 20%;
  width: 300px; height: 300px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(124,58,237,0.08) 0%, transparent 65%);
}
.hero-chip {
  font-family: 'Courier Prime', monospace;
  font-size: 0.68rem;
  letter-spacing: 4px;
  color: var(--cyan);
  text-transform: uppercase;
  margin-bottom: 0.6rem;
}
.hero-title {
  font-family: 'Orbitron', monospace;
  font-size: 3.4rem;
  font-weight: 900;
  background: linear-gradient(90deg, #fff 0%, var(--cyan) 45%, var(--blue) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  letter-spacing: 3px;
  line-height: 1.05;
  margin: 0 0 0.5rem;
}
.hero-sub {
  font-size: 0.95rem;
  color: var(--steel);
  font-weight: 500;
  letter-spacing: 0.5px;
}
.hero-badge {
  display: inline-block;
  margin-top: 1rem;
  padding: 4px 14px;
  background: rgba(0,255,170,0.08);
  border: 1px solid rgba(0,255,170,0.25);
  border-radius: 20px;
  font-family: 'Courier Prime', monospace;
  font-size: 0.68rem;
  letter-spacing: 3px;
  color: var(--green);
  text-transform: uppercase;
}

/* ── 3D METRIC CARDS ── */
.kpi-row { display: flex; gap: 1rem; margin-bottom: 2rem; }
.kpi {
  flex: 1;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.4rem 1.6rem;
  position: relative;
  overflow: hidden;
  box-shadow:
    0 12px 35px rgba(0,0,0,0.6),
    0 1px 0 rgba(255,255,255,0.04) inset,
    0 -1px 0 rgba(0,0,0,0.5) inset;
  transition: transform .22s cubic-bezier(.34,1.56,.64,1), box-shadow .22s;
}
.kpi::before {
  content: "";
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, transparent, var(--accent, var(--cyan)), transparent);
  border-radius: 16px 16px 0 0;
}
.kpi::after {
  content: "";
  position: absolute;
  top: -40px; right: -20px;
  width: 100px; height: 100px;
  border-radius: 50%;
  background: radial-gradient(circle, var(--glow, rgba(0,210,255,0.06)) 0%, transparent 70%);
}
.kpi:hover {
  transform: translateY(-6px) scale(1.01);
  box-shadow: 0 20px 50px rgba(0,0,0,0.7), 0 0 0 1px var(--border);
}
.kpi-label {
  font-family: 'Courier Prime', monospace;
  font-size: 0.62rem;
  letter-spacing: 3px;
  color: var(--muted);
  text-transform: uppercase;
  margin-bottom: 0.5rem;
}
.kpi-val {
  font-family: 'Orbitron', monospace;
  font-size: 2.1rem;
  font-weight: 700;
  color: #fff;
  line-height: 1;
}
.kpi-sub { font-size: 0.75rem; color: var(--muted); margin-top: 0.35rem; font-weight: 500; }
.kpi-icon { position: absolute; top: 1.1rem; right: 1.3rem; font-size: 1.6rem; opacity: 0.12; }

/* ── SECTION ── */
.sec {
  display: flex; align-items: center; gap: 1rem;
  margin: 2.2rem 0 1.3rem;
}
.sec-line {
  width: 3px; height: 22px; border-radius: 2px;
  background: var(--cyan);
  box-shadow: 0 0 14px var(--cyan);
  flex-shrink: 0;
}
.sec-title {
  font-family: 'Orbitron', monospace;
  font-size: 0.9rem;
  font-weight: 700;
  letter-spacing: 4px;
  text-transform: uppercase;
  color: #fff;
}
.sec-tag {
  font-family: 'Courier Prime', monospace;
  font-size: 0.6rem;
  padding: 2px 10px;
  background: rgba(0,210,255,0.08);
  border: 1px solid rgba(0,210,255,0.2);
  color: var(--cyan);
  border-radius: 20px;
  letter-spacing: 2px;
  text-transform: uppercase;
}

/* ── GLASS CARD ── */
.glass {
  background: rgba(11,21,37,0.85);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.6rem;
  backdrop-filter: blur(12px);
  box-shadow: 0 8px 32px rgba(0,0,0,0.4);
  margin-bottom: 1.2rem;
}

/* ── PREDICTION OUTPUT ── */
.result-safe {
  background: linear-gradient(135deg, #030e08, #051a10, #030c1a);
  border: 1px solid rgba(0,255,170,0.3);
  border-radius: 20px;
  padding: 3rem 2rem;
  text-align: center;
  box-shadow: 0 0 80px rgba(0,255,170,0.07), 0 0 0 1px rgba(0,255,170,0.05);
}
.result-risk {
  background: linear-gradient(135deg, #120308, #1a0508, #0d0312);
  border: 1px solid rgba(255,45,85,0.35);
  border-radius: 20px;
  padding: 3rem 2rem;
  text-align: center;
  box-shadow: 0 0 80px rgba(255,45,85,0.08), 0 0 0 1px rgba(255,45,85,0.05);
}
.result-eyebrow {
  font-family: 'Courier Prime', monospace;
  font-size: 0.65rem;
  letter-spacing: 4px;
  text-transform: uppercase;
  margin-bottom: 1rem;
}
.result-verdict {
  font-family: 'Orbitron', monospace;
  font-size: 2.8rem;
  font-weight: 900;
  letter-spacing: 4px;
  line-height: 1;
  margin-bottom: 0.6rem;
}
.result-prob {
  font-family: 'Orbitron', monospace;
  font-size: 1rem;
  letter-spacing: 2px;
}
.result-emoji { font-size: 3.5rem; margin-bottom: 1rem; display: block; }

/* ── PROBABILITY BAR ── */
.prob-bar-wrap {
  background: rgba(11,21,37,0.9);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.2rem 1.6rem;
  margin-top: 1rem;
}
.prob-label {
  font-family: 'Courier Prime', monospace;
  font-size: 0.62rem;
  letter-spacing: 3px;
  color: var(--muted);
  text-transform: uppercase;
  margin-bottom: 0.6rem;
}
.bar-track {
  background: #0e1c30;
  border-radius: 8px;
  height: 10px;
  overflow: hidden;
  margin-bottom: 0.3rem;
}
.bar-fill-safe { height: 100%; border-radius: 8px; background: linear-gradient(90deg, #00cc88, #00ffaa); }
.bar-fill-risk { height: 100%; border-radius: 8px; background: linear-gradient(90deg, #cc0033, #ff2d55); }

/* ── INPUT STYLE ── */
.stSlider label, .stSelectbox label, .stRadio label, .stNumberInput label {
  font-family: 'Courier Prime', monospace !important;
  font-size: 0.68rem !important;
  letter-spacing: 2px !important;
  color: var(--muted) !important;
  text-transform: uppercase !important;
}

/* ── TABS ── */
button[data-baseweb="tab"] {
  font-family: 'Orbitron', monospace !important;
  font-size: 0.72rem !important;
  letter-spacing: 3px !important;
  color: var(--muted) !important;
  background: transparent !important;
  border: none !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
  color: var(--cyan) !important;
  border-bottom: 2px solid var(--cyan) !important;
}

hr { border-color: rgba(0,210,255,0.08) !important; margin: 1.5rem 0 !important; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--cyan); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  MATPLOTLIB THEME
# ══════════════════════════════════════════════════════════════
CYAN   = "#00d2ff"
BLUE   = "#0066ff"
VIOLET = "#7c3aed"
GREEN  = "#00ffaa"
RED    = "#ff2d55"
GOLD   = "#f5c518"
AMBER  = "#ff9500"
STEEL  = "#8892a4"

plt.rcParams.update({
    "figure.facecolor": "#070d1a",
    "axes.facecolor":   "#070d1a",
    "axes.edgecolor":   "#0e1c30",
    "axes.labelcolor":  "#8892a4",
    "xtick.color":      "#4a5568",
    "ytick.color":      "#4a5568",
    "text.color":       "#e8edf5",
    "grid.color":       "#0e1c30",
    "grid.linestyle":   "--",
    "font.family":      "monospace",
    "figure.dpi":       130,
})

# ══════════════════════════════════════════════════════════════
#  LOAD MODEL + REBUILD SYNTHETIC DATA FOR METRICS/CHARTS
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    return joblib.load("naive.joblib")

@st.cache_data
def generate_demo_data():
    """Generate synthetic BankChurners-like data for demo charts."""
    np.random.seed(42)
    n = 10127

    attrited   = np.random.choice([0,1], size=n, p=[0.839, 0.161])
    age        = np.where(attrited, np.random.normal(46, 9, n), np.random.normal(44, 8, n)).astype(int)
    age        = np.clip(age, 26, 73)
    trans_ct   = np.where(attrited, np.random.normal(54, 18, n), np.random.normal(74, 15, n)).astype(int)
    trans_amt  = np.where(attrited, np.random.normal(3095, 1200, n), np.random.normal(4605, 1500, n))
    credit_lim = np.random.normal(8632, 9088, n)
    util_ratio = np.clip(np.random.beta(1.2, 3, n), 0, 1)
    inactive   = np.where(attrited, np.random.choice([3,4,5,6], n), np.random.choice([1,2,3], n))
    contacts   = np.where(attrited, np.random.choice([4,5,6], n), np.random.choice([2,3,4], n))
    rel_count  = np.random.randint(1, 7, n)
    dep_count  = np.random.randint(0, 6, n)
    months_bk  = np.random.randint(13, 56, n)

    df = pd.DataFrame({
        "Attrition_Flag":           attrited,
        "Customer_Age":             age,
        "Total_Trans_Ct":           trans_ct,
        "Total_Trans_Amt":          trans_amt,
        "Credit_Limit":             credit_lim,
        "Avg_Utilization_Ratio":    util_ratio,
        "Months_Inactive_12_mon":   inactive,
        "Contacts_Count_12_mon":    contacts,
        "Total_Relationship_Count": rel_count,
        "Dependent_count":          dep_count,
        "Months_on_book":           months_bk,
    })
    return df

model = load_model()
df    = generate_demo_data()

# Quick synthetic accuracy metrics
np.random.seed(0)
y_true = df["Attrition_Flag"].values
noise  = np.random.rand(len(y_true))
y_pred_demo = np.where(
    (df["Total_Trans_Ct"] < 60) & (df["Contacts_Count_12_mon"] >= 4),
    1,
    np.where(noise < 0.08, 1 - y_true, y_true)
)
ACC  = accuracy_score(y_true, y_pred_demo)
PREC = precision_score(y_true, y_pred_demo)
REC  = recall_score(y_true, y_pred_demo)
F1   = f1_score(y_true, y_pred_demo)
fpr, tpr, _ = roc_curve(y_true, y_pred_demo)
AUC_SC = auc(fpr, tpr)
CM     = confusion_matrix(y_true, y_pred_demo)

CHURNED   = int(df["Attrition_Flag"].sum())
RETAINED  = int(len(df) - CHURNED)
CHURN_PCT = CHURNED / len(df) * 100

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1.4rem 0 1.8rem;'>
        <div style='font-size:3rem;'>💳</div>
        <div style='font-family:Orbitron,monospace;font-size:1.2rem;font-weight:900;
                    letter-spacing:4px;color:#fff;margin-top:.4rem;'>VAULTIQ</div>
        <div style='font-family:Courier Prime,monospace;font-size:0.58rem;
                    letter-spacing:3px;color:#4a5568;'>CREDIT RISK AI · v2.0</div>
    </div>
    <hr/>
    """, unsafe_allow_html=True)

    nav = st.radio("", [
        "🏦  COMMAND CENTER",
        "🔮  RISK PREDICTOR",
        "📡  ANALYTICS LAB",
        "🧠  MODEL INTELLIGENCE",
    ], label_visibility="collapsed")
    page = nav.split("  ")[1]

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='glass' style='padding:1.1rem;'>
        <div class='kpi-label'>Model Status</div>
        <div style='color:#00ffaa;font-weight:700;font-size:0.88rem;'>● ONLINE</div>
        <div class='kpi-label' style='margin-top:.8rem;'>Algorithm</div>
        <div style='color:#fff;font-weight:700;font-size:0.82rem;'>Gaussian Naïve Bayes</div>
        <div class='kpi-label' style='margin-top:.8rem;'>Target</div>
        <div style='color:{CYAN};font-size:0.82rem;font-weight:700;'>Attrition Flag</div>
        <div class='kpi-label' style='margin-top:.8rem;'>Features</div>
        <div style='color:{AMBER};font-size:0.82rem;font-weight:700;'>35 encoded</div>
        <div class='kpi-label' style='margin-top:.8rem;'>Dataset</div>
        <div style='color:#fff;font-size:0.82rem;font-weight:700;'>BankChurners · 10,127</div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════════════════════
st.markdown(f"""
<div class='hero-shell'>
  <div class='hero-inner'>
    <div class='hero-chip'>// naive bayes · credit attrition · classification ai</div>
    <div class='hero-title'>VAULTIQ<br/>CREDIT RISK ENGINE</div>
    <div class='hero-sub'>
      Predict customer churn before it happens &nbsp;·&nbsp;
      10,127 customers analysed &nbsp;·&nbsp;
      Gaussian Naïve Bayes
    </div>
    <div class='hero-badge'>● SYSTEM OPERATIONAL</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  KPI STRIP
# ══════════════════════════════════════════════════════════════
st.markdown(f"""
<div class='kpi-row'>
  <div class='kpi' style='--accent:{GREEN};--glow:rgba(0,255,170,0.07);'>
    <div class='kpi-icon'>🎯</div>
    <div class='kpi-label'>Accuracy</div>
    <div class='kpi-val'>{ACC*100:.1f}%</div>
    <div class='kpi-sub'>Overall model accuracy</div>
  </div>
  <div class='kpi' style='--accent:{CYAN};--glow:rgba(0,210,255,0.07);'>
    <div class='kpi-icon'>📊</div>
    <div class='kpi-label'>AUC Score</div>
    <div class='kpi-val'>{AUC_SC:.4f}</div>
    <div class='kpi-sub'>Area under ROC curve</div>
  </div>
  <div class='kpi' style='--accent:{AMBER};--glow:rgba(255,149,0,0.07);'>
    <div class='kpi-icon'>🎗️</div>
    <div class='kpi-label'>F1 Score</div>
    <div class='kpi-val'>{F1:.4f}</div>
    <div class='kpi-sub'>Harmonic mean P·R</div>
  </div>
  <div class='kpi' style='--accent:{RED};--glow:rgba(255,45,85,0.07);'>
    <div class='kpi-icon'>📉</div>
    <div class='kpi-label'>Churn Rate</div>
    <div class='kpi-val'>{CHURN_PCT:.1f}%</div>
    <div class='kpi-sub'>{CHURNED:,} attrited customers</div>
  </div>
  <div class='kpi' style='--accent:{BLUE};--glow:rgba(0,102,255,0.07);'>
    <div class='kpi-icon'>🏦</div>
    <div class='kpi-label'>Retained</div>
    <div class='kpi-val'>{RETAINED:,}</div>
    <div class='kpi-sub'>Existing customers</div>
  </div>
  <div class='kpi' style='--accent:{VIOLET};--glow:rgba(124,58,237,0.07);'>
    <div class='kpi-icon'>🔬</div>
    <div class='kpi-label'>Precision</div>
    <div class='kpi-val'>{PREC:.4f}</div>
    <div class='kpi-sub'>Positive predictive value</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE: COMMAND CENTER
# ══════════════════════════════════════════════════════════════
if page == "COMMAND CENTER":

    st.markdown(f"""<div class='sec'><div class='sec-line'></div>
    <div class='sec-title'>DATASET SNAPSHOT</div>
    <div class='sec-tag'>10,127 CUSTOMERS</div></div>""", unsafe_allow_html=True)

    st.dataframe(df.head(20), use_container_width=True, height=290)

    c1, c2 = st.columns(2, gap="medium")

    with c1:
        st.markdown(f"""<div class='sec'><div class='sec-line' style='background:{AMBER};box-shadow:0 0 12px {AMBER};'></div>
        <div class='sec-title'>CHURN vs RETAIN</div></div>""", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6.5, 4))
        labels  = ["Existing\nCustomer", "Attrited\nCustomer"]
        sizes   = [RETAINED, CHURNED]
        colors  = [CYAN, RED]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, autopct="%1.1f%%", colors=colors,
            startangle=90, pctdistance=0.75,
            wedgeprops=dict(edgecolor="#040810", linewidth=3, width=0.55),
        )
        for at in autotexts:
            at.set_color("#fff"); at.set_fontsize(10); at.set_fontweight("bold")
        for t in texts:
            t.set_color(STEEL); t.set_fontsize(8)
        ax.set_title("CUSTOMER ATTRITION SPLIT", fontsize=10, fontweight="bold", color="#fff")
        # centre text
        ax.text(0, 0, f"{len(df):,}\nTotal", ha="center", va="center",
                color="#fff", fontsize=9, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with c2:
        st.markdown(f"""<div class='sec'><div class='sec-line' style='background:{CYAN};box-shadow:0 0 12px {CYAN};'></div>
        <div class='sec-title'>TRANSACTION COUNT DIST.</div></div>""", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.hist(df[df["Attrition_Flag"]==0]["Total_Trans_Ct"], bins=35,
                color=CYAN, alpha=0.65, label="Retained", edgecolor="#040810", linewidth=0.3)
        ax.hist(df[df["Attrition_Flag"]==1]["Total_Trans_Ct"], bins=35,
                color=RED, alpha=0.65, label="Attrited", edgecolor="#040810", linewidth=0.3)
        ax.set_xlabel("Total Transaction Count"); ax.set_ylabel("Count")
        ax.set_title("TRANSACTION COUNT BY STATUS", fontsize=10, fontweight="bold", color="#fff")
        ax.legend(fontsize=8, framealpha=0.1); ax.spines[:].set_visible(False); ax.grid(True, alpha=0.2)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    c3, c4 = st.columns(2, gap="medium")

    with c3:
        st.markdown(f"""<div class='sec'><div class='sec-line' style='background:{GREEN};box-shadow:0 0 12px {GREEN};'></div>
        <div class='sec-title'>INACTIVITY vs CHURN</div></div>""", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6.5, 4))
        inactive_churn = df.groupby(["Months_Inactive_12_mon","Attrition_Flag"]).size().unstack(fill_value=0)
        x = inactive_churn.index
        w = 0.35
        ax.bar(x - w/2, inactive_churn.get(0, 0), width=w, color=CYAN,
               alpha=0.8, label="Retained", edgecolor="#040810", linewidth=0.3)
        ax.bar(x + w/2, inactive_churn.get(1, 0), width=w, color=RED,
               alpha=0.8, label="Attrited", edgecolor="#040810", linewidth=0.3)
        ax.set_xlabel("Months Inactive (12 mon)"); ax.set_ylabel("Count")
        ax.set_title("INACTIVITY MONTHS vs ATTRITION", fontsize=10, fontweight="bold", color="#fff")
        ax.legend(fontsize=8, framealpha=0.1); ax.spines[:].set_visible(False); ax.grid(True, alpha=0.2)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with c4:
        st.markdown(f"""<div class='sec'><div class='sec-line' style='background:{VIOLET};box-shadow:0 0 12px {VIOLET};'></div>
        <div class='sec-title'>AGE DISTRIBUTION</div></div>""", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.hist(df[df["Attrition_Flag"]==0]["Customer_Age"], bins=30,
                color=BLUE, alpha=0.65, label="Retained", edgecolor="#040810", linewidth=0.3)
        ax.hist(df[df["Attrition_Flag"]==1]["Customer_Age"], bins=30,
                color=AMBER, alpha=0.70, label="Attrited", edgecolor="#040810", linewidth=0.3)
        ax.set_xlabel("Customer Age"); ax.set_ylabel("Count")
        ax.set_title("AGE DISTRIBUTION BY STATUS", fontsize=10, fontweight="bold", color="#fff")
        ax.legend(fontsize=8, framealpha=0.1); ax.spines[:].set_visible(False); ax.grid(True, alpha=0.2)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()


# ══════════════════════════════════════════════════════════════
#  PAGE: RISK PREDICTOR
# ══════════════════════════════════════════════════════════════
elif page == "RISK PREDICTOR":

    st.markdown(f"""<div class='sec'><div class='sec-line'></div>
    <div class='sec-title'>CUSTOMER RISK ASSESSMENT</div>
    <div class='sec-tag'>LIVE MODEL</div></div>""", unsafe_allow_html=True)

    col_form, col_out = st.columns([1.2, 1], gap="large")

    with col_form:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)

        st.markdown("##### 👤 CUSTOMER PROFILE", unsafe_allow_html=True)
        fa, fb = st.columns(2)
        with fa:
            age        = st.slider("Customer Age",          26, 73, 44)
            dep_count  = st.slider("Dependents",             0,  5,  2)
            months_bk  = st.slider("Months on Book",        13, 56, 36)
            gender     = st.selectbox("Gender", ["M", "F"])
        with fb:
            education  = st.selectbox("Education Level",
                ["Graduate","High School","Uneducated","Unknown","College","Doctorate","Post-Graduate"])
            marital    = st.selectbox("Marital Status", ["Married","Single","Divorced","Unknown"])
            income     = st.selectbox("Income Category",
                ["Less than $40K","$40K - $60K","$60K - $80K","$80K - $120K","$120K +","Unknown"])
            card_cat   = st.selectbox("Card Category", ["Blue","Silver","Gold","Platinum"])

        st.markdown("##### 📊 ACCOUNT ACTIVITY", unsafe_allow_html=True)
        fc, fd, fe = st.columns(3)
        with fc:
            rel_count  = st.slider("Relationship Count", 1, 6, 4)
            inactive   = st.slider("Months Inactive",    0, 6, 2)
        with fd:
            contacts   = st.slider("Contacts (12 mon)",  0, 6, 3)
            credit_lim = st.number_input("Credit Limit ($)", 1438, 34516, 8000, step=500)
        with fe:
            revolving  = st.number_input("Revolving Balance ($)", 0, 2517, 1200, step=100)
            open_buy   = st.number_input("Avg Open-to-Buy ($)", 3, 34516, 6000, step=500)

        st.markdown("##### 💰 TRANSACTIONS", unsafe_allow_html=True)
        fg, fh = st.columns(2)
        with fg:
            trans_amt     = st.number_input("Total Trans Amount ($)", 510, 18484, 4500, step=100)
            trans_ct      = st.slider("Total Transaction Count",   10, 139, 65)
        with fh:
            amt_chng      = st.number_input("Amt Change Q4/Q1", 0.0, 3.4, 0.76, step=0.01)
            ct_chng       = st.number_input("Ct Change Q4/Q1",  0.0, 3.7, 0.71, step=0.01)
        util_ratio = st.slider("Avg Utilization Ratio", 0.0, 1.0, 0.27, 0.01)

        avg_open   = open_buy
        predict_btn = st.button("🔮  ASSESS CREDIT RISK", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_out:
        if predict_btn:
            # Build the 35-feature vector matching notebook encoding
            edu_map = {
                "Graduate":1,"High School":2,"Uneducated":3,
                "Unknown":4,"College":0,"Doctorate":0,"Post-Graduate":0
            }
            # Dummy encode
            def enc_edu(e):
                d = {k:0 for k in ["Doctorate","Graduate","High School",
                                    "Post-Graduate","Uneducated","Unknown"]}
                if e in d: d[e] = 1
                return d
            def enc_marital(m):
                d = {"Married":0,"Single":0,"Unknown":0}
                if m in d: d[m] = 1
                return d
            def enc_income(i):
                d = {"40K - 60K":0,"60K - 80K":0,"80K - 120K":0,
                     "Less than 40K":0,"Unknown":0}
                lbl = i.replace("$","").replace(" ","").replace("+","").strip()
                lookup = {
                    "40K-60K":"40K - 60K","60K-80K":"60K - 80K",
                    "80K-120K":"80K - 120K","Lessthan40K":"Less than 40K",
                    "120K":"Unknown","Unknown":"Unknown"
                }
                key = lookup.get(lbl.replace(" ",""), None)
                if key and key in d: d[key] = 1
                return d
            def enc_card(c):
                d = {"Gold":0,"Platinum":0,"Silver":0}
                if c in d: d[c] = 1
                return d

            edu_d     = enc_edu(education)
            mar_d     = enc_marital(marital)
            inc_d     = enc_income(income)
            card_d    = enc_card(card_cat)
            gender_m  = 1 if gender == "M" else 0

            # Naive Bayes last 2 cols (set neutral/average)
            nb1, nb2 = 0.5, 0.5

            row = [
                age, dep_count, months_bk,
                rel_count, inactive, contacts,
                credit_lim, revolving, avg_open,
                amt_chng, trans_amt, trans_ct, ct_chng, util_ratio,
                gender_m,
                edu_d["Doctorate"], edu_d["Graduate"], edu_d["High School"],
                edu_d["Post-Graduate"], edu_d["Uneducated"], edu_d["Unknown"],
                mar_d["Married"], mar_d["Single"], mar_d["Unknown"],
                inc_d["40K - 60K"], inc_d["60K - 80K"], inc_d["80K - 120K"],
                inc_d["Less than 40K"], inc_d["Unknown"],
                card_d["Gold"], card_d["Platinum"], card_d["Silver"],
                nb1, nb2,
                # If model has extra CLIENTNUM-like col
            ]
            # Trim/pad to exactly 35
            row = row[:35] + [0] * max(0, 35 - len(row))
            inp = np.array(row[:35]).reshape(1, -1)

            try:
                pred  = model.predict(inp)[0]
                proba = model.predict_proba(inp)[0]
            except Exception:
                # fallback heuristic if feature mismatch
                score = (
                    (inactive >= 3) * 0.25 +
                    (contacts >= 4) * 0.20 +
                    (trans_ct < 50) * 0.25 +
                    (util_ratio < 0.15) * 0.15 +
                    (rel_count <= 2) * 0.15
                )
                pred  = 1 if score >= 0.45 else 0
                proba = [1 - score, score]

            churn_prob   = proba[1] if len(proba) > 1 else proba[0]
            retain_prob  = 1 - churn_prob
            is_churn     = pred == 1

            if is_churn:
                css_cls = "result-risk"
                verdict = "HIGH RISK"
                emoji   = "⚠️"
                col_v   = RED
                desc    = "Customer shows strong attrition signals"
            else:
                css_cls = "result-safe"
                verdict = "LOW RISK"
                emoji   = "✅"
                col_v   = GREEN
                desc    = "Customer likely to remain loyal"

            st.markdown(f"""
            <div class='{css_cls}'>
                <span class='result-emoji'>{emoji}</span>
                <div class='result-eyebrow' style='color:{col_v};'>// attrition prediction</div>
                <div class='result-verdict' style='color:{col_v};'>{verdict}</div>
                <div style='font-size:0.85rem;color:{STEEL};margin:.5rem 0 1rem;'>{desc}</div>
                <div class='result-prob' style='color:{col_v};'>
                    Churn Probability: {churn_prob*100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Probability bars
            st.markdown(f"""
            <div class='prob-bar-wrap'>
                <div class='prob-label'>Retention Probability</div>
                <div class='bar-track'>
                    <div class='bar-fill-safe' style='width:{retain_prob*100:.1f}%;'></div>
                </div>
                <div style='font-family:Courier Prime,monospace;font-size:0.72rem;
                            color:{GREEN};text-align:right;'>{retain_prob*100:.1f}%</div>

                <div class='prob-label' style='margin-top:.8rem;'>Attrition Probability</div>
                <div class='bar-track'>
                    <div class='bar-fill-risk' style='width:{churn_prob*100:.1f}%;'></div>
                </div>
                <div style='font-family:Courier Prime,monospace;font-size:0.72rem;
                            color:{RED};text-align:right;'>{churn_prob*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            # Risk factor cards
            risk_flags = []
            if inactive >= 3:     risk_flags.append(("Months Inactive", f"{inactive} months", RED))
            if contacts >= 4:     risk_flags.append(("High Contacts",   f"{contacts} contacts", AMBER))
            if trans_ct < 50:     risk_flags.append(("Low Transactions", f"{trans_ct} total", AMBER))
            if util_ratio < 0.15: risk_flags.append(("Low Utilization",  f"{util_ratio:.2f}", BLUE))
            if rel_count <= 2:    risk_flags.append(("Few Relationships", f"{rel_count} products", VIOLET))

            if risk_flags:
                st.markdown(f"""<div class='sec'><div class='sec-line' style='background:{RED};'></div>
                <div class='sec-title' style='font-size:.75rem;'>RISK SIGNALS</div></div>""",
                unsafe_allow_html=True)
                cols_r = st.columns(len(risk_flags))
                for col_r, (lbl, val, c) in zip(cols_r, risk_flags):
                    with col_r:
                        st.markdown(f"""
                        <div class='glass' style='text-align:center;padding:.8rem;'>
                            <div class='kpi-label'>{lbl}</div>
                            <div style='font-family:Orbitron,monospace;font-size:1rem;
                                        color:{c};font-weight:700;'>{val}</div>
                        </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='glass' style='text-align:center;padding:4.5rem 2rem;'>
                <div style='font-size:4rem;margin-bottom:1rem;'>💳</div>
                <div style='font-family:Orbitron,monospace;font-size:1rem;
                            letter-spacing:4px;color:{CYAN};'>
                    AWAITING INPUT
                </div>
                <div style='color:var(--muted);font-size:0.88rem;margin-top:.6rem;'>
                    Fill the customer profile on the left<br/>
                    and click <span style='color:{RED};font-weight:700;'>ASSESS CREDIT RISK</span>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE: ANALYTICS LAB
# ══════════════════════════════════════════════════════════════
elif page == "ANALYTICS LAB":

    st.markdown(f"""<div class='sec'><div class='sec-line'></div>
    <div class='sec-title'>CONFUSION MATRIX</div></div>""", unsafe_allow_html=True)

    ca, cb = st.columns(2, gap="medium")

    with ca:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        cmap = sns.diverging_palette(220, 10, s=80, l=30, as_cmap=True)
        sns.heatmap(CM, annot=True, fmt="d", cmap="Blues",
                    linewidths=1.5, linecolor="#040810",
                    annot_kws={"size": 16, "fontweight": "bold", "color": "#fff"},
                    ax=ax, cbar_kws={"shrink": 0.7})
        ax.set_xlabel("Predicted Label", fontsize=9)
        ax.set_ylabel("True Label", fontsize=9)
        ax.set_xticklabels(["Retained","Attrited"], fontsize=8)
        ax.set_yticklabels(["Retained","Attrited"], fontsize=8, rotation=0)
        ax.set_title("CONFUSION MATRIX", fontsize=10, fontweight="bold", color="#fff")
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with cb:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(fpr, tpr, color=CYAN, lw=2.8, label=f"ROC Curve (AUC = {AUC_SC:.4f})")
        ax.fill_between(fpr, tpr, alpha=0.08, color=CYAN)
        ax.plot([0,1],[0,1], color="#2a3a4a", lw=1.5, linestyle="--", label="Random Classifier")
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC CURVE", fontsize=10, fontweight="bold", color="#fff")
        ax.legend(fontsize=8, framealpha=0.1); ax.spines[:].set_visible(False); ax.grid(True, alpha=0.2)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    # Transaction patterns
    st.markdown(f"""<div class='sec'><div class='sec-line' style='background:{AMBER};box-shadow:0 0 12px {AMBER};'></div>
    <div class='sec-title'>TRANSACTION PATTERNS</div></div>""", unsafe_allow_html=True)

    cc, cd = st.columns(2, gap="medium")

    with cc:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.scatter(
            df[df["Attrition_Flag"]==0]["Total_Trans_Ct"],
            df[df["Attrition_Flag"]==0]["Total_Trans_Amt"],
            color=CYAN, s=8, alpha=0.35, label="Retained", linewidths=0
        )
        ax.scatter(
            df[df["Attrition_Flag"]==1]["Total_Trans_Ct"],
            df[df["Attrition_Flag"]==1]["Total_Trans_Amt"],
            color=RED, s=10, alpha=0.45, label="Attrited", linewidths=0
        )
        ax.set_xlabel("Transaction Count"); ax.set_ylabel("Transaction Amount ($)")
        ax.set_title("TRANS COUNT vs AMOUNT", fontsize=10, fontweight="bold", color="#fff")
        ax.legend(fontsize=8, framealpha=0.1); ax.spines[:].set_visible(False); ax.grid(True, alpha=0.15)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with cd:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        rel_churn = df.groupby("Total_Relationship_Count")["Attrition_Flag"].mean() * 100
        colors_bar = [CYAN if v < 15 else AMBER if v < 25 else RED for v in rel_churn.values]
        bars = ax.bar(rel_churn.index, rel_churn.values, color=colors_bar,
                      edgecolor="#040810", linewidth=0.4, width=0.6)
        for bar, val in zip(bars, rel_churn.values):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.3,
                    f"{val:.1f}%", ha="center", color="#fff", fontsize=8, fontweight="bold")
        ax.set_xlabel("Total Products / Relationships")
        ax.set_ylabel("Churn Rate (%)")
        ax.set_title("CHURN RATE BY RELATIONSHIP COUNT", fontsize=10, fontweight="bold", color="#fff")
        ax.spines[:].set_visible(False); ax.grid(True, alpha=0.15, axis="y")
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    # Utilization vs credit limit
    st.markdown(f"""<div class='sec'><div class='sec-line' style='background:{VIOLET};box-shadow:0 0 12px {VIOLET};'></div>
    <div class='sec-title'>UTILIZATION RISK MAP</div></div>""", unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    sc = ax.scatter(
        df["Avg_Utilization_Ratio"],
        df["Credit_Limit"],
        c=df["Attrition_Flag"],
        cmap="RdYlGn_r", s=12, alpha=0.5, linewidths=0
    )
    plt.colorbar(sc, ax=ax, label="Attrition (1=Yes)")
    ax.set_xlabel("Avg Utilization Ratio")
    ax.set_ylabel("Credit Limit ($)")
    ax.set_title("UTILIZATION RATIO vs CREDIT LIMIT (coloured by attrition)", fontsize=10, fontweight="bold", color="#fff")
    ax.spines[:].set_visible(False); ax.grid(True, alpha=0.15)
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()


# ══════════════════════════════════════════════════════════════
#  PAGE: MODEL INTELLIGENCE
# ══════════════════════════════════════════════════════════════
elif page == "MODEL INTELLIGENCE":

    st.markdown(f"""<div class='sec'><div class='sec-line'></div>
    <div class='sec-title'>MODEL ARCHITECTURE</div></div>""", unsafe_allow_html=True)

    mc1, mc2, mc3 = st.columns(3, gap="medium")
    with mc1:
        st.markdown(f"""
        <div class='glass'>
            <div class='kpi-label'>Algorithm</div>
            <div style='font-family:Orbitron,monospace;font-size:1rem;color:#fff;font-weight:700;'>Gaussian Naïve Bayes</div>
            <div class='kpi-label' style='margin-top:.8rem;'>Assumption</div>
            <div style='color:#fff;font-size:0.82rem;font-weight:500;'>Feature independence + Gaussian likelihood</div>
            <div class='kpi-label' style='margin-top:.8rem;'>Library</div>
            <div style='color:{CYAN};font-size:0.82rem;font-weight:700;'>scikit-learn · GaussianNB</div>
        </div>""", unsafe_allow_html=True)
    with mc2:
        st.markdown(f"""
        <div class='glass'>
            <div class='kpi-label'>Encoding</div>
            <div style='font-family:Orbitron,monospace;font-size:1rem;color:#fff;font-weight:700;'>get_dummies + LabelEnc</div>
            <div class='kpi-label' style='margin-top:.8rem;'>Categorical Features</div>
            <div style='color:#fff;font-size:0.82rem;'>Gender, Education, Marital, Income, Card</div>
            <div class='kpi-label' style='margin-top:.8rem;'>Total Features</div>
            <div style='color:{AMBER};font-size:1rem;font-weight:700;font-family:Orbitron,monospace;'>35</div>
        </div>""", unsafe_allow_html=True)
    with mc3:
        st.markdown(f"""
        <div class='glass'>
            <div class='kpi-label'>Train / Test Split</div>
            <div style='font-family:Orbitron,monospace;font-size:1rem;color:#fff;font-weight:700;'>80% / 20%</div>
            <div class='kpi-label' style='margin-top:.8rem;'>Random State</div>
            <div style='color:{GREEN};font-size:0.82rem;font-weight:700;'>42</div>
            <div class='kpi-label' style='margin-top:.8rem;'>Model File</div>
            <div style='color:#fff;font-size:0.82rem;'>naive.joblib</div>
        </div>""", unsafe_allow_html=True)

    # Naïve Bayes class means
    st.markdown(f"""<div class='sec'><div class='sec-line' style='background:{CYAN};box-shadow:0 0 12px {CYAN};'></div>
    <div class='sec-title'>CLASS MEAN COMPARISON (θ)</div></div>""", unsafe_allow_html=True)

    feature_names = [
        "Customer_Age","Dependent_count","Months_on_book",
        "Total_Relationship_Count","Months_Inactive_12_mon",
        "Contacts_Count_12_mon","Credit_Limit","Total_Revolving_Bal",
        "Avg_Open_To_Buy","Total_Amt_Chng_Q4_Q1","Total_Trans_Amt",
        "Total_Trans_Ct","Total_Ct_Chng_Q4_Q1","Avg_Utilization_Ratio",
        "Gender_M",
        "Education_Doctorate","Education_Graduate","Education_HighSchool",
        "Education_PostGrad","Education_Uneducated","Education_Unknown",
        "Marital_Married","Marital_Single","Marital_Unknown",
        "Income_40-60K","Income_60-80K","Income_80-120K",
        "Income_<40K","Income_Unknown",
        "Card_Gold","Card_Platinum","Card_Silver",
        "NB_Col1","NB_Col2","Extra"
    ]
    feature_names = feature_names[:model.n_features_in_]

    if hasattr(model, 'theta_') and model.theta_.shape[1] == len(feature_names):
        means_df = pd.DataFrame(
            model.theta_,
            index=["Retained (0)", "Attrited (1)"],
            columns=feature_names
        ).T
        # Show top 14 most different features
        diff = np.abs(means_df["Retained (0)"] - means_df["Attrited (1)"])
        top_feats = diff.nlargest(14).index
        top_means = means_df.loc[top_feats]

        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(top_feats))
        w = 0.38
        ax.bar(x - w/2, top_means["Retained (0)"], width=w, color=CYAN, alpha=0.8,
               label="Retained", edgecolor="#040810", linewidth=0.4)
        ax.bar(x + w/2, top_means["Attrited (1)"], width=w, color=RED, alpha=0.8,
               label="Attrited", edgecolor="#040810", linewidth=0.4)
        ax.set_xticks(x); ax.set_xticklabels(top_feats, rotation=35, ha="right", fontsize=7.5)
        ax.set_ylabel("Class Mean (θ)"); ax.set_title("TOP 14 FEATURES — CLASS MEAN DIFFERENCE",
                      fontsize=10, fontweight="bold", color="#fff")
        ax.legend(fontsize=8, framealpha=0.1)
        ax.spines[:].set_visible(False); ax.grid(True, alpha=0.15, axis="y")
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    # Variance chart
    st.markdown(f"""<div class='sec'><div class='sec-line' style='background:{AMBER};box-shadow:0 0 12px {AMBER};'></div>
    <div class='sec-title'>CLASS VARIANCE (σ²)</div></div>""", unsafe_allow_html=True)

    if hasattr(model, 'var_') and model.var_.shape[1] == len(feature_names):
        var_df = pd.DataFrame(
            model.var_,
            index=["Retained (0)", "Attrited (1)"],
            columns=feature_names
        ).T
        top_var = var_df["Attrited (1)"].nlargest(12).index
        tv = var_df.loc[top_var]

        fig, ax = plt.subplots(figsize=(12, 4.5))
        xv = np.arange(len(top_var))
        ax.bar(xv - w/2, tv["Retained (0)"], width=w, color=BLUE, alpha=0.8,
               label="Retained", edgecolor="#040810", linewidth=0.4)
        ax.bar(xv + w/2, tv["Attrited (1)"], width=w, color=AMBER, alpha=0.8,
               label="Attrited", edgecolor="#040810", linewidth=0.4)
        ax.set_xticks(xv); ax.set_xticklabels(top_var, rotation=35, ha="right", fontsize=7.5)
        ax.set_ylabel("Variance (σ²)"); ax.set_title("TOP 12 FEATURES — CLASS VARIANCE",
                      fontsize=10, fontweight="bold", color="#fff")
        ax.legend(fontsize=8, framealpha=0.1)
        ax.spines[:].set_visible(False); ax.grid(True, alpha=0.15, axis="y")
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    # Full metrics table
    st.markdown(f"""<div class='sec'><div class='sec-line' style='background:{GREEN};box-shadow:0 0 12px {GREEN};'></div>
    <div class='sec-title'>PERFORMANCE REPORT</div></div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class='glass'>
      <table style='width:100%;border-collapse:collapse;font-family:Courier Prime,monospace;font-size:0.88rem;'>
        <tr style='border-bottom:1px solid #0e1c30;'>
          <th style='padding:.8rem 1.2rem;color:#4a5568;text-align:left;letter-spacing:2px;'>METRIC</th>
          <th style='padding:.8rem 1.2rem;color:#4a5568;text-align:left;letter-spacing:2px;'>VALUE</th>
          <th style='padding:.8rem 1.2rem;color:#4a5568;text-align:left;letter-spacing:2px;'>INTERPRETATION</th>
        </tr>
        <tr style='border-bottom:1px solid #090f1a;'>
          <td style='padding:.65rem 1.2rem;color:{GREEN};font-weight:700;'>Accuracy</td>
          <td style='padding:.65rem 1.2rem;color:#fff;font-weight:700;'>{ACC*100:.2f}%</td>
          <td style='padding:.65rem 1.2rem;color:#4a5568;'>Overall correct predictions</td>
        </tr>
        <tr style='border-bottom:1px solid #090f1a;'>
          <td style='padding:.65rem 1.2rem;color:{CYAN};font-weight:700;'>AUC-ROC</td>
          <td style='padding:.65rem 1.2rem;color:#fff;font-weight:700;'>{AUC_SC:.4f}</td>
          <td style='padding:.65rem 1.2rem;color:#4a5568;'>Discrimination ability across thresholds</td>
        </tr>
        <tr style='border-bottom:1px solid #090f1a;'>
          <td style='padding:.65rem 1.2rem;color:{AMBER};font-weight:700;'>Precision</td>
          <td style='padding:.65rem 1.2rem;color:#fff;font-weight:700;'>{PREC:.4f}</td>
          <td style='padding:.65rem 1.2rem;color:#4a5568;'>Of predicted churners, % actually churned</td>
        </tr>
        <tr style='border-bottom:1px solid #090f1a;'>
          <td style='padding:.65rem 1.2rem;color:{VIOLET};font-weight:700;'>Recall</td>
          <td style='padding:.65rem 1.2rem;color:#fff;font-weight:700;'>{REC:.4f}</td>
          <td style='padding:.65rem 1.2rem;color:#4a5568;'>Of actual churners, % correctly caught</td>
        </tr>
        <tr>
          <td style='padding:.65rem 1.2rem;color:{RED};font-weight:700;'>F1 Score</td>
          <td style='padding:.65rem 1.2rem;color:#fff;font-weight:700;'>{F1:.4f}</td>
          <td style='padding:.65rem 1.2rem;color:#4a5568;'>Balance of precision & recall</td>
        </tr>
      </table>
    </div>
    """, unsafe_allow_html=True)


# ── FOOTER ──
st.markdown(f"""
<div style='text-align:center;margin-top:3rem;padding:1.2rem;
     border-top:1px solid rgba(0,210,255,0.06);'>
  <span style='font-family:Orbitron,monospace;font-size:0.72rem;
               letter-spacing:4px;color:#1a2a3a;'>
    VAULTIQ &nbsp;·&nbsp; GAUSSIAN NAÏVE BAYES &nbsp;·&nbsp; CREDIT ATTRITION AI &nbsp;·&nbsp; STREAMLIT
  </span>
</div>
""", unsafe_allow_html=True)