"""
app.py  –  Board Game User Segmentation & Revenue Analysis
Interactive Streamlit Dashboard

Run with:
    streamlit run app.py

Prerequisites:
    1. pip install -r requirements.txt
    2. python train_and_save_models.py   (trains + saves models to models/)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import streamlit as st
import joblib

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BGG Analytics",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour palette ─────────────────────────────────────────────────────────────
PALETTE = ["#4C72B0", "#DD8452", "#55A868"]   # seaborn default first 3

# ── Helper: load artefact (cached) ────────────────────────────────────────────
MODELS_DIR = "models"

@st.cache_data(show_spinner=False)
def load_artifact(filename):
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        return None
    ext = os.path.splitext(filename)[1]
    if ext == ".parquet":
        return pd.read_parquet(path)
    return joblib.load(path)

def models_ready():
    required = [
        "kmeans_model.pkl", "kmeans_scaler.pkl", "cluster_profiles.pkl",
        "demand_model.pkl", "demand_scaler.pkl", "market_sample.pkl",
        "games_summary.parquet", "ratings_sample.pkl",
    ]
    return all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in required)

# ── Sidebar navigation ─────────────────────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/BoardGameGeek-logo.svg/320px-BoardGameGeek-logo.svg.png",
    use_container_width=True,
)
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["📊 EDA Dashboard", "👥 User Segments", "🔮 Demand Predictor"],
    index=0,
)
st.sidebar.markdown("---")
st.sidebar.caption("Board Game User Segmentation\n& Revenue Analysis")

# ── Guard: check models exist ──────────────────────────────────────────────────
if not models_ready():
    st.error(
        "⚠️  Model files not found in `models/`.\n\n"
        "Please run the training script first:\n"
        "```\npython train_and_save_models.py\n```"
    )
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EDA Dashboard
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 EDA Dashboard":
    st.title("📊 EDA Dashboard")
    st.markdown(
        "Explore rating distributions and market penetration of board games "
        "from the **BoardGameGeek** dataset."
    )

    # Load artefacts
    ratings_sample  = load_artifact("ratings_sample.pkl")
    df_games        = load_artifact("games_summary.parquet")

    # ── Sidebar filters ────────────────────────────────────────────────────────
    st.sidebar.subheader("Filters")
    rating_range = st.sidebar.slider(
        "Rating range", min_value=1.0, max_value=10.0,
        value=(1.0, 10.0), step=0.5
    )
    min_voters = st.sidebar.number_input(
        "Min. voters", min_value=0, max_value=50_000, value=0, step=500
    )

    # Filter ratings sample
    mask_rating = (ratings_sample >= rating_range[0]) & (ratings_sample <= rating_range[1])
    filtered_ratings = ratings_sample[mask_rating]

    # Filter games
    df_plot = df_games.copy()
    if "usersrated" in df_plot.columns:
        df_plot = df_plot[df_plot["usersrated"] >= min_voters]

    # ── Row 1: histograms ──────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("User Rating Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(filtered_ratings, bins=30, kde=True, color=PALETTE[0], ax=ax)
        ax.set_xlabel("Rating (1 – 10)")
        ax.set_ylabel("Count")
        ax.set_title(f"n = {len(filtered_ratings):,} ratings")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Market Penetration (Log Owners)")
        if "owned" in df_plot.columns:
            valid = df_plot[df_plot["owned"] > 0]["owned"]
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.histplot(np.log1p(valid), bins=30, kde=True, color=PALETTE[1], ax=ax2)
            ax2.set_xlabel("log(1 + Total Owners)")
            ax2.set_ylabel("Number of Games")
            ax2.set_title(f"n = {len(valid):,} games")
            st.pyplot(fig2)
            plt.close(fig2)
        else:
            st.info("'owned' column not available in the dataset.")

    st.markdown("---")

    # ── Row 2: Top games ───────────────────────────────────────────────────────
    st.subheader("🏆 Top 15 Games by Market Demand (Owners)")
    if "owned" in df_plot.columns and "primary" in df_plot.columns:
        top = (
            df_plot.dropna(subset=["owned", "primary"])
            .sort_values("owned", ascending=False)
            .head(15)
        )
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        sns.barplot(data=top, x="owned", y="primary", palette="Blues_r", ax=ax3)
        ax3.set_xlabel("Total Owners")
        ax3.set_ylabel("")
        ax3.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)
    else:
        st.info("Game name / owners data not available.")

    # ── Summary stats ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📋 Summary Statistics")
    if not df_plot.empty:
        num_cols = [c for c in ["average", "averageweight", "usersrated", "owned"]
                    if c in df_plot.columns]
        st.dataframe(df_plot[num_cols].describe().round(2), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — User Segments
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥 User Segments":
    st.title("👥 User Segments")
    st.markdown(
        "Three distinct user personas identified via **K-Means clustering** "
        "on review behaviour features."
    )

    cluster_profiles = load_artifact("cluster_profiles.pkl")

    if cluster_profiles is None:
        st.error("Cluster profiles not found. Run `python train_and_save_models.py` first.")
        st.stop()

    # ── Persona cards ──────────────────────────────────────────────────────────
    persona_icons = {"Quality Seekers": "⭐", "Power Reviewers": "✍️", "Casual Raters": "🎲"}
    persona_desc  = {
        "Quality Seekers":  "High average ratings, moderate activity – they only review games they love.",
        "Power Reviewers":  "Most reviews per user; engaged enthusiasts covering a wide range of games.",
        "Casual Raters":    "Lower review count; come in with high ratings – satisfied but infrequent.",
    }

    cols = st.columns(3)
    for i, (_, row) in enumerate(cluster_profiles.iterrows()):
        persona = row.get("Persona", f"Cluster {i}")
        with cols[i]:
            st.metric(label=f"{persona_icons.get(persona, '🎯')} {persona}",
                      value=f"{int(row['User_Count']):,} users")
            st.caption(persona_desc.get(persona, ""))
            st.markdown(
                f"- Avg. Rating: **{row['avg_rating']:.2f}**\n"
                f"- Avg. Reviews/User: **{row['num_ratings']:.1f}**\n"
                f"- Std. Dev. Rating: **{row['std_rating']:.3f}**"
            )

    st.markdown("---")

    # ── Cluster profiles table ─────────────────────────────────────────────────
    st.subheader("Cluster Profiles Table")
    display_df = cluster_profiles.copy()
    display_df.index.name = "Cluster"
    st.dataframe(display_df, use_container_width=True)

    st.markdown("---")

    # ── Radar / grouped bar chart ──────────────────────────────────────────────
    st.subheader("Feature Comparison Across Clusters")

    features = ["num_ratings", "avg_rating", "std_rating"]
    labels   = ["Reviews / User", "Avg. Rating", "Std. Dev. Rating"]

    # Normalise to 0-1 for visual balance
    norm_df = cluster_profiles[features].copy()
    for col in features:
        col_min, col_max = norm_df[col].min(), norm_df[col].max()
        if col_max > col_min:
            norm_df[col] = (norm_df[col] - col_min) / (col_max - col_min)

    fig4, ax4 = plt.subplots(figsize=(8, 4))
    x = np.arange(len(features))
    width = 0.25
    for i, (idx, row) in enumerate(norm_df.iterrows()):
        persona = cluster_profiles.loc[idx, "Persona"] if "Persona" in cluster_profiles.columns else f"Cluster {idx}"
        ax4.bar(x + i * width, row[features].values, width, label=persona, color=PALETTE[i % len(PALETTE)])
    ax4.set_xticks(x + width)
    ax4.set_xticklabels(labels)
    ax4.set_ylabel("Normalised Score (0 – 1)")
    ax4.set_title("Relative Feature Strength per Segment")
    ax4.legend()
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)

    st.markdown("---")

    # ── Business interpretation ────────────────────────────────────────────────
    st.subheader("💡 Business Interpretation")
    st.markdown("""
| Segment | Revenue Strategy |
|---|---|
| ⭐ Quality Seekers | Target with premium editions and expansions for beloved games |
| ✍️ Power Reviewers | Leverage as influencers; early-access review copies drive word-of-mouth |
| 🎲 Casual Raters | Broad appeal; introductory products and bundle deals |
""")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Demand Predictor
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Demand Predictor":
    st.title("🔮 Demand Predictor")
    st.markdown(
        "Predict **market demand** (estimated number of owners) for a board game "
        "using the trained **Linear Regression** model."
    )

    lr_model      = load_artifact("demand_model.pkl")
    scaler_lr     = load_artifact("demand_scaler.pkl")
    market_sample = load_artifact("market_sample.pkl")

    if lr_model is None or scaler_lr is None:
        st.error("Model files not found. Run `python train_and_save_models.py` first.")
        st.stop()

    # ── Input form ─────────────────────────────────────────────────────────────
    st.subheader("Game Attributes")
    col_l, col_r = st.columns([1, 1])

    with col_l:
        avg_rating = st.slider(
            "⭐ Average User Rating", min_value=1.0, max_value=10.0,
            value=7.0, step=0.1,
            help="Community average rating on BoardGameGeek (1 = worst, 10 = best)"
        )
        complexity = st.slider(
            "🧩 Game Complexity / Weight", min_value=1.0, max_value=5.0,
            value=2.5, step=0.1,
            help="BGG weight rating (1 = light family game, 5 = heavy strategy game)"
        )

    with col_r:
        num_voters = st.number_input(
            "🗳️ Number of Voters (users who rated)", min_value=0,
            max_value=200_000, value=5_000, step=500,
            help="How many BGG users have submitted a rating for this game"
        )
        category = st.selectbox(
            "📦 Game Category",
            ["Strategy", "Family", "Party", "Thematic / Adventure",
             "War Game", "Abstract", "Children's Game", "Other"],
            help="Broad genre (informational only – not used in the regression model)"
        )

    st.markdown("---")

    # ── Predict ────────────────────────────────────────────────────────────────
    if st.button("🔍 Predict Demand", type="primary"):
        raw_input = np.array([[avg_rating, complexity, num_voters]])
        scaled    = scaler_lr.transform(raw_input)
        log_pred  = lr_model.predict(scaled)[0]
        pred_owners = int(np.expm1(log_pred))   # reverse log1p

        st.markdown("### 📈 Prediction Result")
        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.metric("Predicted Market Demand", f"{pred_owners:,} owners")
        with res_col2:
            # Percentile among market sample
            pct = int(np.mean(market_sample <= pred_owners) * 100)
            st.metric("Market Percentile", f"Top {100 - pct}%")
        with res_col3:
            tier = (
                "🟢 High Demand"   if pct >= 75 else
                "🟡 Mid Demand"    if pct >= 40 else
                "🔴 Niche / Low"
            )
            st.metric("Demand Tier", tier)

        # ── Market comparison chart ────────────────────────────────────────────
        st.markdown("#### How Does This Game Compare to the Market?")
        fig5, ax5 = plt.subplots(figsize=(9, 4))
        sns.histplot(
            np.log1p(market_sample), bins=40, kde=True,
            color=PALETTE[0], alpha=0.6, ax=ax5, label="Market Distribution"
        )
        ax5.axvline(
            x=log_pred, color="#E74C3C", linewidth=2.5,
            linestyle="--", label=f"Your Game ({pred_owners:,} owners)"
        )
        ax5.set_xlabel("log(1 + Owners)")
        ax5.set_ylabel("Number of Games")
        ax5.set_title("Predicted Demand vs. Market Distribution")
        ax5.legend()
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close(fig5)

        # ── Explanation ────────────────────────────────────────────────────────
        with st.expander("ℹ️ How is this calculated?"):
            st.markdown(f"""
**Model:** Linear Regression trained on ~{len(market_sample):,} board games from BoardGameGeek.

**Features used:**
| Feature | Your Input |
|---|---|
| Average User Rating | {avg_rating} |
| Game Complexity / Weight | {complexity} |
| Number of Voters | {num_voters:,} |

**Note:** The target variable (`owned`) was log-transformed during training to handle
the right-skewed distribution. The predicted value is then exponentiated back to the
original scale.  Category (`{category}`) is displayed but not included in the model.
""")
    else:
        st.info("👈  Adjust the sliders and click **Predict Demand** to see results.")
