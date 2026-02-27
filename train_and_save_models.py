"""
train_and_save_models.py
------------------------
Run this script ONCE (after placing the dataset in DataSet/archive/) to train
and persist the K-Means and Linear Regression models used by the Streamlit app.

Usage:
    python train_and_save_models.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
REVIEWS_PATH = "DataSet/archive/bgg-15m-reviews.csv"
GAMES_PATH   = "DataSet/archive/games_detailed_info.csv"
MODELS_DIR   = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data …")
try:
    df_reviews = pd.read_csv(REVIEWS_PATH)  # load entire dataset
except FileNotFoundError:
    raise FileNotFoundError(
        f"Reviews file not found at '{REVIEWS_PATH}'. "
        "Please download the dataset from Kaggle and place it in DataSet/archive/."
    )

try:
    df_games = pd.read_csv(GAMES_PATH)
    if "id" in df_games.columns:
        df_games.rename(columns={"id": "ID"}, inplace=True)
except FileNotFoundError:
    raise FileNotFoundError(
        f"Games metadata file not found at '{GAMES_PATH}'."
    )

print(f"  Reviews : {df_reviews.shape}")
print(f"  Games   : {df_games.shape}")

# ── Clean reviews ─────────────────────────────────────────────────────────────
print("Cleaning reviews …")
df_reviews.dropna(subset=["rating", "user"], inplace=True)
df_reviews.drop_duplicates(subset=["user", "ID"], inplace=True)

# ── K-Means: user segmentation ────────────────────────────────────────────────
print("Training K-Means (user segmentation) …")
user_features = (
    df_reviews.groupby("user")["rating"]
    .agg(num_ratings="count", avg_rating="mean", std_rating="std")
    .dropna()
    .reset_index()
)

scaler_km = StandardScaler()
X_km = scaler_km.fit_transform(user_features[["num_ratings", "avg_rating", "std_rating"]])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_km)
user_features["Cluster"] = kmeans.labels_

cluster_profiles = (
    user_features.groupby("Cluster")[["num_ratings", "avg_rating", "std_rating"]]
    .mean()
    .round(3)
)
cluster_profiles["User_Count"] = user_features.groupby("Cluster").size()

# Give each cluster a human-readable persona name based on characteristics
profiles_sorted = cluster_profiles.sort_values("avg_rating", ascending=False)
persona_map = {}
for rank, idx in enumerate(profiles_sorted.index):
    if rank == 0:
        persona_map[idx] = "Quality Seekers"
    elif rank == 1:
        persona_map[idx] = "Power Reviewers"
    else:
        persona_map[idx] = "Casual Raters"
cluster_profiles["Persona"] = cluster_profiles.index.map(persona_map)

print(cluster_profiles)

# ── Linear Regression: demand prediction ─────────────────────────────────────
print("Training Linear Regression (demand prediction) …")

# Feature engineering on games metadata:
#   averageweight  → complexity
#   average        → avg community rating
#   users_rated    → number of voters
#   owned          → target (proxy for market demand)
LR_COLS = ["average", "averageweight", "usersrated"]
TARGET   = "owned"

df_lr = df_games[LR_COLS + [TARGET]].dropna()
df_lr = df_lr[df_lr[TARGET] > 0]

X_lr_raw = df_lr[LR_COLS].values
y_lr      = np.log1p(df_lr[TARGET].values)   # log-transform skewed target

scaler_lr = StandardScaler()
X_lr = scaler_lr.fit_transform(X_lr_raw)

X_train, X_test, y_train, y_test = train_test_split(
    X_lr, y_lr, test_size=0.2, random_state=42
)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred = lr_model.predict(X_test)
print(f"  R²  : {r2_score(y_test, y_pred):.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}  (log scale)")

# Keep a small sample of raw 'owned' values for the market chart in the app
market_sample = df_lr[TARGET].sample(min(5000, len(df_lr)), random_state=42).values

# ── Save artefacts ────────────────────────────────────────────────────────────
print("Saving models …")
joblib.dump(kmeans,           os.path.join(MODELS_DIR, "kmeans_model.pkl"))
joblib.dump(scaler_km,        os.path.join(MODELS_DIR, "kmeans_scaler.pkl"))
joblib.dump(cluster_profiles, os.path.join(MODELS_DIR, "cluster_profiles.pkl"))
joblib.dump(lr_model,         os.path.join(MODELS_DIR, "demand_model.pkl"))
joblib.dump(scaler_lr,        os.path.join(MODELS_DIR, "demand_scaler.pkl"))
joblib.dump(market_sample,    os.path.join(MODELS_DIR, "market_sample.pkl"))

# Also save a small games summary for EDA page (avoid loading 21 k-row CSV each time)
eda_cols = ["primary", "average", "averageweight", "usersrated", "owned",
            "yearpublished", "minplayers", "maxplayers"]
available = [c for c in eda_cols if c in df_games.columns]
df_games_small = df_games[available].copy()
df_games_small.to_parquet(os.path.join(MODELS_DIR, "games_summary.parquet"), index=False)

# Save a small reviews rating sample for EDA
ratings_sample = df_reviews["rating"].dropna().sample(
    min(500_000, len(df_reviews)), random_state=42
).values  # 500 k sample – enough for EDA charts
joblib.dump(ratings_sample, os.path.join(MODELS_DIR, "ratings_sample.pkl"))

print("Done! All files saved to the 'models/' directory.")
print("You can now launch the app with:  streamlit run app.py")
