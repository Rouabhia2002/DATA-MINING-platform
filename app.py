# ============================================================
#  DATA MINING STUDIO — Streamlit Application
#  Author  : Senior Data Scientist
#  Stack   : Streamlit · Pandas · NumPy · Sklearn · Seaborn
#  Run     : streamlit run app.py
# ============================================================

import io
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    silhouette_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Data Mining Studio",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────────────
#  GLOBAL CUSTOM CSS  — dark-teal industrial palette
# ────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Import fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

    /* ── Root palette ── */
    :root {
        --bg:        #0d1117;
        --surface:   #161b22;
        --surface2:  #1c2230;
        --accent:    #00c9a7;
        --accent2:   #0078d4;
        --warn:      #f0a500;
        --danger:    #e05c5c;
        --text:      #e6edf3;
        --muted:     #7d8fa3;
        --border:    #30363d;
        --radius:    10px;
    }

    /* ── Base ── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: var(--bg) !important;
        color: var(--text) !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: var(--surface) !important;
        border-right: 1px solid var(--border);
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent), var(--accent2));
        color: #000 !important;
        font-weight: 600;
        border: none;
        border-radius: var(--radius);
        padding: 0.5rem 1.4rem;
        transition: opacity .2s, transform .1s;
    }
    .stButton > button:hover { opacity:.88; transform:translateY(-1px); }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: var(--surface2);
        border: 1px solid var(--border);
        border-left: 3px solid var(--accent);
        border-radius: var(--radius);
        padding: 1rem 1.2rem;
    }

    /* ── DataFrames ── */
    [data-testid="stDataFrame"] { border-radius: var(--radius); }

    /* ── Tabs ── */
    [data-baseweb="tab-list"] {
        gap: 4px;
        background: var(--surface);
        border-radius: var(--radius);
        padding: 4px;
    }
    [data-baseweb="tab"] {
        border-radius: 8px !important;
        font-weight: 500;
    }
    [aria-selected="true"] {
        background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
        color: #000 !important;
    }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        background: var(--surface2);
        border: 1px solid var(--border);
        border-radius: var(--radius);
    }

    /* ── Section card helper ── */
    .card {
        background: var(--surface2);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
    }

    /* ── Hero banner ── */
    .hero {
        background: linear-gradient(135deg, #0d1117 0%, #0f2027 50%, #101e2b 100%);
        border: 1px solid var(--border);
        border-left: 4px solid var(--accent);
        border-radius: var(--radius);
        padding: 1.8rem 2rem;
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        font-family: 'Space Mono', monospace;
        font-size: 2rem;
        color: var(--accent) !important;
        margin: 0 0 .4rem 0;
    }
    .hero p { color: var(--muted); margin: 0; font-size: .95rem; }

    /* ── Badge ── */
    .badge {
        display: inline-block;
        background: rgba(0,201,167,.15);
        color: var(--accent);
        border: 1px solid rgba(0,201,167,.3);
        border-radius: 20px;
        padding: 2px 10px;
        font-size: .78rem;
        font-weight: 600;
        margin-right: 4px;
    }

    /* ── Divider ── */
    .divider {
        border: none;
        border-top: 1px solid var(--border);
        margin: 1.5rem 0;
    }

    /* ── Score pill ── */
    .score-pill {
        display: inline-block;
        background: linear-gradient(135deg, var(--accent), var(--accent2));
        color: #000;
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        font-size: 1.6rem;
        border-radius: 40px;
        padding: .3rem 1.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ────────────────────────────────────────────────────────────
#  SESSION STATE INITIALISATION
# ────────────────────────────────────────────────────────────
_defaults = {
    "raw_df":       None,   # Original uploaded dataframe
    "clean_df":     None,   # After missing-value treatment
    "processed_df": None,   # After normalisation
    "target_col":   None,   # Target column for classification
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ────────────────────────────────────────────────────────────
#  HELPERS
# ────────────────────────────────────────────────────────────
def styled_header(icon: str, title: str, subtitle: str = "") -> None:
    """Render a consistent section header."""
    st.markdown(
        f"""
        <div style="margin-bottom:1rem;">
          <span style="font-size:1.6rem">{icon}</span>
          <span style="font-family:'Space Mono',monospace;font-size:1.25rem;
                       font-weight:700;color:#00c9a7;margin-left:.5rem;">{title}</span>
          {"<br><span style='color:#7d8fa3;font-size:.9rem;margin-left:2rem;'>" + subtitle + "</span>" if subtitle else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def divider() -> None:
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def fig_to_streamlit(fig, caption: str = "") -> None:
    """Render a matplotlib figure with transparent background."""
    fig.patch.set_facecolor("#1c2230")
    for ax in fig.axes:
        ax.set_facecolor("#1c2230")
        ax.tick_params(colors="#7d8fa3")
        ax.xaxis.label.set_color("#7d8fa3")
        ax.yaxis.label.set_color("#7d8fa3")
        ax.title.set_color("#e6edf3")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
    if caption:
        st.pyplot(fig, caption=caption)
    else:
        st.pyplot(fig)
    plt.close(fig)


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# ────────────────────────────────────────────────────────────
#  SIDEBAR NAVIGATION
# ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center;padding:1rem 0 1.5rem 0;">
          <div style="font-family:'Space Mono',monospace;font-size:1.1rem;
                      color:#00c9a7;font-weight:700;">🔬 DATA MINING</div>
          <div style="font-family:'Space Mono',monospace;font-size:.7rem;
                      color:#7d8fa3;letter-spacing:3px;">STUDIO v1.0</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    nav = st.radio(
        "Navigation",
        [
            "🏠  Home",
            "⚙️  Preprocessing",
            "🔵  Clustering",
            "🤖  Classification",
        ],
        label_visibility="collapsed",
    )

    divider()

    # ── Sidebar dataset status
    st.markdown("**Dataset Status**")
    if st.session_state.raw_df is not None:
        df_info = st.session_state.processed_df if st.session_state.processed_df is not None else st.session_state.raw_df
        st.success(f"✅ Loaded — {df_info.shape[0]} rows × {df_info.shape[1]} cols")
        if st.session_state.target_col:
            st.info(f"🎯 Target: `{st.session_state.target_col}`")
    else:
        st.warning("⚠️ No dataset loaded yet")

    divider()
    st.markdown(
        "<small style='color:#7d8fa3;'>Built with Streamlit · sklearn<br>© 2025 Data Mining Studio</small>",
        unsafe_allow_html=True,
    )

page = nav.split("  ", 1)[-1]   # strip icon prefix


# ════════════════════════════════════════════════════════════
#  HOME PAGE
# ════════════════════════════════════════════════════════════
if page == "Home":
    st.markdown(
        """
        <div class="hero">
          <h1>🔬 Data Mining Studio</h1>
          <p>An end-to-end data mining pipeline — preprocessing, clustering, and classification —
             wrapped in a clean, professional interface.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="card">
              <div style="font-size:2rem">⚙️</div>
              <div style="font-family:'Space Mono',monospace;font-weight:700;
                          color:#00c9a7;margin:.4rem 0 .2rem;">Preprocessing</div>
              <div style="color:#7d8fa3;font-size:.88rem;">
                Upload CSV · Explore · Clean missing values ·
                Normalize · Visualize boxplots & scatter plots
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="card">
              <div style="font-size:2rem">🔵</div>
              <div style="font-family:'Space Mono',monospace;font-weight:700;
                          color:#00c9a7;margin:.4rem 0 .2rem;">Clustering</div>
              <div style="color:#7d8fa3;font-size:.88rem;">
                K-Means · K-Medoids · Elbow method ·
                Silhouette score · PCA 2-D cluster plot
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="card">
              <div style="font-size:2rem">🤖</div>
              <div style="font-family:'Space Mono',monospace;font-weight:700;
                          color:#00c9a7;margin:.4rem 0 .2rem;">Classification</div>
              <div style="color:#7d8fa3;font-size:.88rem;">
                Logistic Regression · KNN · Decision Tree ·
                Confusion Matrix · Accuracy · F1-score
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    divider()
    st.markdown("### 🚀 Quick-start guide")
    st.markdown(
        """
        1. Navigate to **⚙️ Preprocessing** in the sidebar and upload your CSV file.
        2. Explore, clean, and normalise your data.
        3. Move to **🔵 Clustering** to discover natural groupings.
        4. Head to **🤖 Classification** to train and evaluate predictive models.
        """
    )
    st.info(
        "💡 **Tip:** All processed data is stored in session state and shared across modules automatically."
    )


# ════════════════════════════════════════════════════════════
#  PREPROCESSING PAGE
# ════════════════════════════════════════════════════════════
elif page == "Preprocessing":

    st.markdown(
        "<h2 style='font-family:Space Mono,monospace;color:#00c9a7;'>⚙️ Preprocessing</h2>",
        unsafe_allow_html=True,
    )

    # ── TABS inside the page
    t1, t2, t3, t4, t5 = st.tabs(
        ["📥 Import", "🔍 Explore", "🧹 Clean", "📏 Normalise", "📊 Visualise"]
    )

    # ── 1. IMPORT ────────────────────────────────────────────
    with t1:
        styled_header("📥", "Data Import", "Upload a CSV dataset to begin")

        uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded:
            with st.spinner("Loading dataset …"):
                df = load_csv(uploaded.getvalue())
                st.session_state.raw_df = df.copy()
                st.session_state.clean_df = df.copy()
                st.session_state.processed_df = None  # reset on new upload

            st.success(f"✅ Dataset loaded — **{df.shape[0]} rows × {df.shape[1]} columns**")
            divider()
            st.markdown("**Preview (first 10 rows)**")
            st.dataframe(df.head(10), use_container_width=True)

        elif st.session_state.raw_df is not None:
            df = st.session_state.raw_df
            st.info("✅ Dataset already in memory. Re-upload to replace it.")
            st.dataframe(df.head(10), use_container_width=True)
        else:
            st.markdown(
                "<div class='card' style='text-align:center;color:#7d8fa3;padding:2.5rem;'>"
                "📂 No file uploaded yet. Use the uploader above.</div>",
                unsafe_allow_html=True,
            )

    # ── 2. EXPLORE ───────────────────────────────────────────
    with t2:
        styled_header("🔍", "Data Exploration", "Understand the shape and quality of your data")

        df = st.session_state.raw_df
        if df is None:
            st.warning("⚠️ Please upload a dataset first (Import tab).")
        else:
            # Shape + dtypes row
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", f"{df.shape[0]:,}")
            c2.metric("Columns", df.shape[1])
            c3.metric("Numeric cols", len(df.select_dtypes(include=np.number).columns))
            c4.metric(
                "Missing cells",
                f"{df.isnull().sum().sum():,}",
                delta=f"{df.isnull().mean().mean()*100:.1f}% of data",
                delta_color="inverse",
            )

            divider()
            c_left, c_right = st.columns(2)

            with c_left:
                with st.expander("📋 Data Types", expanded=True):
                    dtype_df = pd.DataFrame(
                        {"Column": df.dtypes.index, "Type": df.dtypes.values.astype(str)}
                    )
                    st.dataframe(dtype_df, use_container_width=True, hide_index=True)

            with c_right:
                with st.expander("❌ Missing Values", expanded=True):
                    mv = df.isnull().sum()
                    mv_pct = (mv / len(df) * 100).round(2)
                    mv_df = pd.DataFrame({"Missing": mv, "Pct (%)": mv_pct})
                    mv_df = mv_df[mv_df["Missing"] > 0]
                    if mv_df.empty:
                        st.success("🎉 No missing values found!")
                    else:
                        st.dataframe(mv_df, use_container_width=True)

            divider()
            with st.expander("📊 Summary Statistics", expanded=False):
                st.dataframe(df.describe().T.round(4), use_container_width=True)

    # ── 3. CLEAN ─────────────────────────────────────────────
    with t3:
        styled_header("🧹", "Data Cleaning", "Handle missing values before modelling")

        df = st.session_state.raw_df
        if df is None:
            st.warning("⚠️ Please upload a dataset first (Import tab).")
        else:
            missing_total = df.isnull().sum().sum()
            if missing_total == 0:
                st.success("🎉 Your dataset has no missing values — nothing to clean!")
                st.session_state.clean_df = df.copy()
            else:
                st.info(f"Found **{missing_total}** missing cell(s) across **{(df.isnull().sum()>0).sum()}** column(s).")
                strategy = st.selectbox(
                    "Missing-value strategy",
                    ["Drop rows with any missing value",
                     "Fill with Mean  (numeric cols)",
                     "Fill with Median (numeric cols)",
                     "Fill with Mode  (all cols)"],
                )

                if st.button("🧹 Apply Cleaning"):
                    with st.spinner("Cleaning …"):
                        cleaned = df.copy()
                        if "Drop" in strategy:
                            cleaned = cleaned.dropna()
                        elif "Mean" in strategy:
                            num = cleaned.select_dtypes(include=np.number).columns
                            cleaned[num] = cleaned[num].fillna(cleaned[num].mean())
                        elif "Median" in strategy:
                            num = cleaned.select_dtypes(include=np.number).columns
                            cleaned[num] = cleaned[num].fillna(cleaned[num].median())
                        else:  # Mode
                            for col in cleaned.columns:
                                cleaned[col] = cleaned[col].fillna(cleaned[col].mode()[0])

                        st.session_state.clean_df = cleaned
                        st.session_state.processed_df = None  # reset normalisation

                    n_removed = df.shape[0] - cleaned.shape[0]
                    st.success(
                        f"✅ Cleaning done — remaining: **{cleaned.shape[0]} rows × {cleaned.shape[1]} cols**"
                        + (f" ({n_removed} rows removed)" if n_removed > 0 else "")
                    )
                    st.dataframe(cleaned.head(8), use_container_width=True)

                # Download cleaned data
                if st.session_state.clean_df is not None:
                    st.download_button(
                        "⬇️ Download Cleaned Dataset",
                        data=df_to_csv_bytes(st.session_state.clean_df),
                        file_name="cleaned_dataset.csv",
                        mime="text/csv",
                    )

    # ── 4. NORMALISE ─────────────────────────────────────────
    with t4:
        styled_header("📏", "Normalisation", "Scale numeric features before modelling")

        clean_df = st.session_state.clean_df
        if clean_df is None:
            st.warning("⚠️ Clean your data first (Clean tab).")
        else:
            num_cols = clean_df.select_dtypes(include=np.number).columns.tolist()
            if not num_cols:
                st.error("No numeric columns found.")
            else:
                c1, c2 = st.columns([2, 1])
                with c1:
                    method = st.selectbox(
                        "Normalisation method",
                        ["Min-Max Scaling  (0–1 range)",
                         "Standardisation  (zero mean, unit variance)"],
                    )
                    cols_to_scale = st.multiselect(
                        "Columns to normalise", num_cols, default=num_cols
                    )
                with c2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(
                        """
                        <div class="card">
                          <b>Min-Max</b><br>
                          <code style="color:#00c9a7;">x' = (x−min)/(max−min)</code><br><br>
                          <b>StandardScaler</b><br>
                          <code style="color:#00c9a7;">x' = (x−μ)/σ</code>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                if st.button("📏 Apply Normalisation") and cols_to_scale:
                    with st.spinner("Scaling …"):
                        proc = clean_df.copy()
                        if "Min-Max" in method:
                            scaler = MinMaxScaler()
                        else:
                            scaler = StandardScaler()
                        proc[cols_to_scale] = scaler.fit_transform(proc[cols_to_scale])
                        st.session_state.processed_df = proc

                    st.success("✅ Normalisation applied!")
                    st.dataframe(proc.head(8), use_container_width=True)

                    # Before / after stats comparison
                    divider()
                    st.markdown("**Before vs After — descriptive stats**")
                    b_col, a_col = st.columns(2)
                    with b_col:
                        st.caption("Before")
                        st.dataframe(
                            clean_df[cols_to_scale].describe().round(4),
                            use_container_width=True,
                        )
                    with a_col:
                        st.caption("After")
                        st.dataframe(
                            proc[cols_to_scale].describe().round(4),
                            use_container_width=True,
                        )

                if st.session_state.processed_df is not None:
                    st.download_button(
                        "⬇️ Download Normalised Dataset",
                        data=df_to_csv_bytes(st.session_state.processed_df),
                        file_name="normalised_dataset.csv",
                        mime="text/csv",
                    )

    # ── 5. VISUALISE ─────────────────────────────────────────
    with t5:
        styled_header("📊", "Visualisation", "Explore feature distributions and relationships")

        df_viz = (
            st.session_state.processed_df
            if st.session_state.processed_df is not None
            else st.session_state.clean_df
            if st.session_state.clean_df is not None
            else st.session_state.raw_df
        )

        if df_viz is None:
            st.warning("⚠️ Please upload a dataset first.")
        else:
            num_cols = df_viz.select_dtypes(include=np.number).columns.tolist()
            all_cols = df_viz.columns.tolist()

            v1, v2 = st.columns(2)

            # ── Boxplot
            with v1:
                st.markdown("#### 📦 Boxplot")
                box_cols = st.multiselect(
                    "Select columns", num_cols,
                    default=num_cols[:min(5, len(num_cols))],
                    key="box_cols",
                )
                if box_cols:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    df_viz[box_cols].plot.box(
                        ax=ax,
                        patch_artist=True,
                        boxprops=dict(facecolor="#00c9a7", color="#00c9a7", alpha=.4),
                        medianprops=dict(color="#f0a500", linewidth=2),
                        whiskerprops=dict(color="#7d8fa3"),
                        capprops=dict(color="#7d8fa3"),
                        flierprops=dict(
                            marker="o", markerfacecolor="#e05c5c",
                            markersize=4, alpha=.6
                        ),
                    )
                    ax.set_title("Feature Boxplot", fontsize=12)
                    plt.xticks(rotation=30, ha="right", fontsize=9)
                    plt.tight_layout()
                    fig_to_streamlit(fig)

            # ── Scatter plot
            with v2:
                st.markdown("#### 🔵 Scatter Plot")
                if len(num_cols) >= 2:
                    x_col = st.selectbox("X-axis", num_cols, index=0, key="scatter_x")
                    y_col = st.selectbox("Y-axis", num_cols, index=1, key="scatter_y")
                    hue_col = st.selectbox(
                        "Colour by (optional)", ["None"] + all_cols, key="scatter_hue"
                    )
                    fig, ax = plt.subplots(figsize=(6, 4))
                    hue = None if hue_col == "None" else df_viz[hue_col]
                    scatter = ax.scatter(
                        df_viz[x_col], df_viz[y_col],
                        c=(
                            pd.factorize(hue)[0]
                            if hue is not None else None
                        ),
                        cmap="cool", alpha=0.7, s=25, edgecolors="none",
                    )
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    ax.set_title(f"{x_col}  vs  {y_col}", fontsize=12)
                    if hue is not None:
                        plt.colorbar(scatter, ax=ax, label=hue_col)
                    plt.tight_layout()
                    fig_to_streamlit(fig)
                else:
                    st.info("Need at least 2 numeric columns for scatter plot.")

            # ── Correlation heatmap (bonus)
            divider()
            with st.expander("🌡️ Correlation Heatmap", expanded=False):
                if len(num_cols) > 1:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    corr = df_viz[num_cols].corr()
                    mask = np.triu(np.ones_like(corr, dtype=bool))
                    sns.heatmap(
                        corr, mask=mask, ax=ax,
                        cmap="coolwarm", center=0,
                        annot=len(num_cols) <= 12,
                        fmt=".2f", linewidths=.5,
                        cbar_kws={"shrink": .8},
                    )
                    ax.set_title("Pearson Correlation Matrix", fontsize=13)
                    plt.tight_layout()
                    fig_to_streamlit(fig)
                else:
                    st.info("Not enough numeric columns for correlation matrix.")


# ════════════════════════════════════════════════════════════
#  CLUSTERING PAGE
# ════════════════════════════════════════════════════════════
elif page == "Clustering":

    st.markdown(
        "<h2 style='font-family:Space Mono,monospace;color:#00c9a7;'>🔵 Clustering</h2>",
        unsafe_allow_html=True,
    )

    # Active dataframe
    df_c = (
        st.session_state.processed_df
        if st.session_state.processed_df is not None
        else st.session_state.clean_df
        if st.session_state.clean_df is not None
        else st.session_state.raw_df
    )

    if df_c is None:
        st.warning("⚠️ Please upload and preprocess a dataset first.")
        st.stop()

    num_cols = df_c.select_dtypes(include=np.number).columns.tolist()
    if len(num_cols) < 2:
        st.error("Need at least 2 numeric columns for clustering.")
        st.stop()

    # ── Feature selection
    with st.expander("🔧 Feature Selection", expanded=True):
        sel_cols = st.multiselect(
            "Select features for clustering",
            num_cols, default=num_cols,
        )

    if not sel_cols:
        st.info("Select at least 2 features above.")
        st.stop()

    X = df_c[sel_cols].dropna().values

    # ── Parameters
    c_left, c_right = st.columns([1, 2])
    with c_left:
        st.markdown("#### ⚙️ Parameters")
        algo = st.selectbox("Algorithm", ["K-Means", "K-Medoids"])
        k = st.slider("Number of clusters (k)", 2, 12, 3)
        random_state = 42

    # ── Elbow Method
    with c_right:
        st.markdown("#### 📈 Elbow Method")
        with st.spinner("Computing inertia for k = 2 … 10 …"):
            inertias = []
            ks_range = range(2, 11)
            for ki in ks_range:
                km = KMeans(n_clusters=ki, random_state=random_state, n_init=10)
                km.fit(X)
                inertias.append(km.inertia_)

        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(list(ks_range), inertias, "o-", color="#00c9a7", linewidth=2, markersize=6)
        ax.axvline(k, color="#f0a500", linestyle="--", linewidth=1.5, label=f"Selected k={k}")
        ax.set_xlabel("Number of clusters k")
        ax.set_ylabel("Inertia (WCSS)")
        ax.set_title("Elbow Method")
        ax.legend(fontsize=9)
        plt.tight_layout()
        fig_to_streamlit(fig)

    divider()

    # ── Run clustering
    if st.button("🚀 Run Clustering", type="primary"):
        with st.spinner(f"Running {algo} with k={k} …"):

            if algo == "K-Means":
                model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
                labels = model.fit_predict(X)
            else:
                # K-Medoids via sklearn_extra; graceful fallback
                try:
                    from sklearn_extra.cluster import KMedoids
                    model = KMedoids(n_clusters=k, random_state=random_state)
                    labels = model.fit_predict(X)
                except ImportError:
                    st.warning(
                        "⚠️ `sklearn-extra` not installed. Falling back to K-Means. "
                        "Install via: `pip install scikit-learn-extra`"
                    )
                    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
                    labels = model.fit_predict(X)

            sil = silhouette_score(X, labels)

            # PCA for 2-D plot
            pca = PCA(n_components=2, random_state=random_state)
            X2d = pca.fit_transform(X)

        # ── Metrics
        st.markdown("#### 📊 Evaluation")
        col1, col2, col3 = st.columns(3)
        col1.metric("Algorithm", algo)
        col2.metric("Clusters (k)", k)
        col3.metric(
            "Silhouette Score",
            f"{sil:.4f}",
            help="Ranges −1 to +1. Closer to +1 = well-separated clusters.",
        )

        divider()

        # ── Cluster plot
        st.markdown("#### 📉 Cluster Visualisation (PCA 2D)")
        PALETTE = [
            "#00c9a7","#0078d4","#f0a500","#e05c5c",
            "#a259ff","#ff6b6b","#4ecdc4","#45b7d1",
            "#96ceb4","#feca57","#ff9ff3","#54a0ff",
        ]
        colors = [PALETTE[l % len(PALETTE)] for l in labels]

        fig, ax = plt.subplots(figsize=(8, 5))
        scatter = ax.scatter(
            X2d[:, 0], X2d[:, 1],
            c=labels, cmap="tab10", alpha=0.75, s=30, edgecolors="none",
        )
        # Cluster centres in PCA space
        if hasattr(model, "cluster_centers_"):
            centers_2d = pca.transform(model.cluster_centers_)
            ax.scatter(
                centers_2d[:, 0], centers_2d[:, 1],
                s=200, c="white", marker="*",
                edgecolors="#f0a500", linewidths=1.5,
                zorder=5, label="Centroid",
            )
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
        ax.set_title(f"{algo} — k={k}  |  Silhouette={sil:.4f}")
        patches = [
            mpatches.Patch(color=PALETTE[i % len(PALETTE)], label=f"Cluster {i}")
            for i in range(k)
        ]
        ax.legend(handles=patches, loc="upper right", fontsize=8, framealpha=.3)
        plt.tight_layout()
        fig_to_streamlit(fig)

        # ── Cluster size distribution
        divider()
        with st.expander("📋 Cluster Composition", expanded=False):
            df_clustered = df_c[sel_cols].dropna().copy()
            df_clustered["Cluster"] = labels
            cluster_counts = df_clustered["Cluster"].value_counts().sort_index()

            fig2, ax2 = plt.subplots(figsize=(6, 3))
            bars = ax2.bar(
                [f"C{i}" for i in cluster_counts.index],
                cluster_counts.values,
                color=[PALETTE[i % len(PALETTE)] for i in cluster_counts.index],
                edgecolor="none", width=0.55,
            )
            for bar in bars:
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    str(bar.get_height()), ha="center", va="bottom",
                    fontsize=9, color="#e6edf3",
                )
            ax2.set_ylabel("Count")
            ax2.set_title("Points per Cluster")
            plt.tight_layout()
            fig_to_streamlit(fig2)

            st.dataframe(
                df_clustered.groupby("Cluster")[sel_cols].mean().round(4),
                use_container_width=True,
            )


# ════════════════════════════════════════════════════════════
#  CLASSIFICATION PAGE
# ════════════════════════════════════════════════════════════
elif page == "Classification":

    st.markdown(
        "<h2 style='font-family:Space Mono,monospace;color:#00c9a7;'>🤖 Classification</h2>",
        unsafe_allow_html=True,
    )

    df_cl = (
        st.session_state.processed_df
        if st.session_state.processed_df is not None
        else st.session_state.clean_df
        if st.session_state.clean_df is not None
        else st.session_state.raw_df
    )

    if df_cl is None:
        st.warning("⚠️ Please upload and preprocess a dataset first.")
        st.stop()

    num_cols = df_cl.select_dtypes(include=np.number).columns.tolist()
    all_cols = df_cl.columns.tolist()

    # ── Setup panel
    with st.expander("🔧 Setup — Target & Features", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            target = st.selectbox(
                "🎯 Target column (label)",
                all_cols,
                index=len(all_cols) - 1,
                key="target_select",
            )
            st.session_state.target_col = target
        with c2:
            feature_pool = [c for c in num_cols if c != target]
            features = st.multiselect(
                "📋 Feature columns (numeric)",
                feature_pool,
                default=feature_pool,
            )

        test_size = st.slider("Test set size (%)", 10, 40, 20, step=5)
        algo_choice = st.selectbox(
            "🤖 Algorithm",
            ["Logistic Regression", "K-Nearest Neighbours (KNN)", "Decision Tree"],
        )

    if not features:
        st.info("Select at least one feature column.")
        st.stop()

    # ── Advanced options
    with st.expander("⚙️ Algorithm Hyperparameters", expanded=False):
        if "KNN" in algo_choice:
            n_neighbors = st.slider("n_neighbors", 1, 25, 5)
        elif "Decision Tree" in algo_choice:
            max_depth = st.slider("max_depth (None = unlimited)", 1, 20, 5)
        else:
            max_iter = st.slider("max_iter", 100, 2000, 500, step=100)
            c_val = st.number_input("C (regularisation strength)", 0.001, 100.0, 1.0, step=0.5)

    divider()

    if st.button("🚀 Train & Evaluate", type="primary"):
        # ── Prepare data
        try:
            df_model = df_cl[features + [target]].dropna()
            X = df_model[features].values
            y = df_model[target].values
        except Exception as e:
            st.error(f"Data preparation failed: {e}")
            st.stop()

        if len(np.unique(y)) < 2:
            st.error("Target column must have at least 2 unique classes.")
            st.stop()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size / 100, random_state=42, stratify=y
        )

        # ── Train
        with st.spinner(f"Training {algo_choice} …"):
            if "Logistic" in algo_choice:
                clf = LogisticRegression(
                    C=c_val, max_iter=max_iter, random_state=42
                )
            elif "KNN" in algo_choice:
                clf = KNeighborsClassifier(n_neighbors=n_neighbors)
            else:
                clf = DecisionTreeClassifier(
                    max_depth=None if max_depth == 20 else max_depth,
                    random_state=42,
                )
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

        # ── Metrics
        acc   = accuracy_score(y_test, y_pred)
        prec  = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec   = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1    = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        st.markdown("#### 📊 Evaluation Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy",  f"{acc*100:.2f}%")
        m2.metric("Precision", f"{prec*100:.2f}%")
        m3.metric("Recall",    f"{rec*100:.2f}%")
        m4.metric("F1-score",  f"{f1*100:.2f}%")

        divider()

        # ── Confusion matrix + classification report
        cm_col, rep_col = st.columns([1, 1])

        with cm_col:
            st.markdown("#### 🗃️ Confusion Matrix")
            classes = np.unique(y)
            cm = confusion_matrix(y_test, y_pred, labels=classes)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                cm, annot=True, fmt="d", ax=ax,
                cmap=sns.light_palette("#00c9a7", as_cmap=True),
                xticklabels=classes, yticklabels=classes,
                linewidths=.5, linecolor="#30363d",
                cbar_kws={"shrink": .8},
            )
            ax.set_xlabel("Predicted", fontsize=10)
            ax.set_ylabel("Actual", fontsize=10)
            ax.set_title("Confusion Matrix", fontsize=12)
            plt.xticks(rotation=30, ha="right", fontsize=9)
            plt.yticks(rotation=0, fontsize=9)
            plt.tight_layout()
            fig_to_streamlit(fig)

        with rep_col:
            st.markdown("#### 📋 Classification Report")
            report = classification_report(
                y_test, y_pred, output_dict=True, zero_division=0
            )
            report_df = pd.DataFrame(report).T.round(3)
            # Style high values green
            st.dataframe(
                report_df.style.background_gradient(
                    cmap="YlGn", subset=["precision","recall","f1-score"]
                ),
                use_container_width=True,
            )

        divider()

        # ── Train/Test split details
        with st.expander("🔍 Dataset Split Details", expanded=False):
            s1, s2, s3 = st.columns(3)
            s1.metric("Total samples", len(X))
            s2.metric("Train samples", len(X_train))
            s3.metric("Test samples",  len(X_test))
            st.caption(
                f"Train: {100-test_size}% · Test: {test_size}% · "
                f"Features: {len(features)} · Classes: {len(classes)}"
            )

        # ── Feature importances (Decision Tree)
        if "Decision Tree" in algo_choice and hasattr(clf, "feature_importances_"):
            with st.expander("🌳 Feature Importances", expanded=False):
                fi = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=True)
                fig, ax = plt.subplots(figsize=(6, max(3, len(features)*0.35)))
                fi.plot.barh(
                    ax=ax, color="#00c9a7", edgecolor="none", alpha=0.85
                )
                ax.set_title("Feature Importances (Gini)", fontsize=12)
                ax.set_xlabel("Importance")
                plt.tight_layout()
                fig_to_streamlit(fig)
