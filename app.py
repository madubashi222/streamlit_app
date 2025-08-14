#!/usr/bin/env python3
import os
import json
import pickle
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Titanic Survival — ML & Streamlit", page_icon="🚢", layout="wide")

# ------------------------------
# Paths & Constants
# ------------------------------
DATA_PATHS = [
    "data/dataset.csv",     # recommended name in this project
    "data/train.csv",       # common Kaggle file name
    "dataset.csv",          # fallback
    "train.csv",            # fallback
]
MODEL_PATH = "model.pkl"
METRICS_PATH = "artifacts/metrics.json"

TARGET_COL = "Survived"
CAT_COLS = ["Sex", "Pclass", "Embarked"]
NUM_COLS = ["Age", "SibSp", "Parch", "Fare"]

ALL_FEATURES = CAT_COLS + NUM_COLS

HELP_DATA_TEXT = """
**Dataset expected**: Titanic training data with columns:

- Survived (0/1), Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
- Optional columns ignored if present: Name, Ticket, Cabin, PassengerId

**How to add**:
1) Put your CSV at `data/dataset.csv` (recommended), or
2) Use the uploader below to load the file at runtime.
"""

# ------------------------------
# Utils
# ------------------------------
@st.cache_data(show_spinner=False)
def load_data_from_disk() -> pd.DataFrame:
    for p in DATA_PATHS:
        if os.path.exists(p):
            df = pd.read_csv(p)
            return df
    return None

@st.cache_data(show_spinner=False)
def load_data_from_upload(upload) -> pd.DataFrame:
    return pd.read_csv(upload)

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure needed columns exist
    missing = [c for c in ALL_FEATURES + [TARGET_COL] if c not in df.columns]
    if missing:
        st.warning(f"Dataset is missing required columns: {missing}")
    return df

def build_preprocess():
    numeric_tf = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])

    categorical_tf = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, NUM_COLS),
            ("cat", categorical_tf, CAT_COLS)
        ]
    )
    return pre

def make_models():
    models = {
        "LogisticRegression": Pipeline([
            ("pre", build_preprocess()),
            ("clf", LogisticRegression(max_iter=1000))
        ]),
        "RandomForest": Pipeline([
            ("pre", build_preprocess()),
            ("clf", RandomForestClassifier(n_estimators=300, random_state=42))
        ]),
        "SVM": Pipeline([
            ("pre", build_preprocess()),
            ("clf", SVC(probability=True, kernel="rbf", C=1.0, gamma="scale", random_state=42))
        ])
    }
    return models

def train_and_select_model(df: pd.DataFrame):
    X = df[ALL_FEATURES].copy()
    y = df[TARGET_COL].astype(int).copy()

    models = make_models()
    scores = {}

    with st.spinner("Training and cross-validating models..."):
        for name, pipe in models.items():
            acc = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
            scores[name] = float(np.mean(acc))

        # pick best
        best_name = max(scores, key=scores.get)
        best_model = models[best_name]
        best_model.fit(X, y)

        # hold-out split for a confusion matrix & report
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        acc_holdout = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        report = classification_report(y_test, y_pred, output_dict=True)

        # Save model + metrics
        joblib.dump(best_model, MODEL_PATH)
        metrics = {
            "cv_scores": scores,
            "best_model": best_name,
            "holdout_accuracy": float(acc_holdout),
            "classification_report": report,
            "confusion_matrix": cm.tolist()
        }
        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    return best_model, metrics

def load_model_and_metrics():
    model = None
    metrics = None
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
        except Exception as e:
            st.warning(f"Couldn't load model.pkl: {e}")
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception as e:
            st.warning(f"Couldn't load metrics: {e}")
    return model, metrics

def ensure_model(df: pd.DataFrame):
    model, metrics = load_model_and_metrics()
    if model is None:
        model, metrics = train_and_select_model(df)
    return model, metrics

def probability_from_model(model, X_df: pd.DataFrame):
    try:
        proba = model.predict_proba(X_df)[:, 1]
        return proba
    except Exception:
        # SVM or others may not have calibrated probabilities; fallback to 0/1
        preds = model.predict(X_df)
        return preds.astype(float)

# ------------------------------
# UI — Sidebar
# ------------------------------
with st.sidebar:
    st.title("🚢 Titanic ML App")
    section = st.radio("Go to", ["Overview", "Explore Data", "Visualizations", "Model & Prediction", "Performance"], index=0)
    st.markdown("---")
    st.caption("Tip: Put your CSV at `data/dataset.csv` for auto-load.")

# ------------------------------
# Data Loading
# ------------------------------
df_disk = load_data_from_disk()

uploaded = None
if df_disk is None:
    with st.sidebar:
        st.info("No dataset found on disk.")
        uploaded = st.file_uploader("Upload Titanic CSV", type=["csv"])

if df_disk is not None:
    df = df_disk.copy()
elif uploaded is not None:
    df = load_data_from_upload(uploaded)
else:
    df = None

if df is not None:
    df = clean_columns(df)
    # Trim to the needed columns + target if present
    keep_cols = [c for c in ALL_FEATURES + [TARGET_COL] if c in df.columns]
    df = df[keep_cols + [c for c in df.columns if c not in keep_cols]]  # keep extra cols at the end for browsing

# ------------------------------
# Pages
# ------------------------------
if section == "Overview":
    st.title("Titanic Survival Prediction — Streamlit App")
    st.write("Interactive EDA, model training, prediction, and evaluation.")
   # 🌊 Custom sea-style background and Titanic header
    st.markdown("""
      <style>
        html, body, [data-testid="stApp"] {
            background: linear-gradient(to bottom, #e0f7fa, #b2ebf2);
        }
        .stButton>button {
            background-color: #0288d1;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 0.5rem 1rem;
        }
        .stButton>button:hover {
            background-color: #01579b;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <img src="https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg" width="250" style="border-radius: 5px;" />
        <h1 style="color: #01579b; margin-top: 0.5rem;">Titanic Survival Prediction</h1>
        <p style="color: #0277bd; font-size: 1.1rem;">Explore the passenger data, visualize survival trends, and try out ML-powered predictions — all aboard the Titanic.</p>
    </div>
""", unsafe_allow_html=True)

   

elif section == "Explore Data":
    st.header("Dataset Overview")
    if df is None:
        st.error("Please upload or add a dataset first.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", len(df))
        col2.metric("Columns", len(df.columns))
        col3.metric("Target present", "Yes" if TARGET_COL in df.columns else "No")

        with st.expander("Show sample data"):
            st.dataframe(df.head(20), use_container_width=True)

        with st.expander("Column types"):
            st.write(df.dtypes.astype(str))

        st.subheader("Interactive Filter")
        cols = st.columns(3)
        sex = cols[0].multiselect("Sex", sorted([x for x in df["Sex"].dropna().unique()]) if "Sex" in df.columns else [], default=None)
        pclass = cols[1].multiselect("Pclass", sorted([int(x) for x in df["Pclass"].dropna().unique()]) if "Pclass" in df.columns else [], default=None)
        embarked = cols[2].multiselect("Embarked", sorted([x for x in df["Embarked"].dropna().unique()]) if "Embarked" in df.columns else [], default=None)

        filtered = df.copy()
        if sex:
            filtered = filtered[filtered["Sex"].isin(sex)]
        if pclass:
            filtered = filtered[filtered["Pclass"].isin(pclass)]
        if embarked:
            filtered = filtered[filtered["Embarked"].isin(embarked)]

        st.write(f"Filtered rows: {len(filtered)}")
        st.dataframe(filtered.head(50), use_container_width=True)

elif section == "Visualizations":
    st.header("Visualizations")
    if df is None:
        st.error("Please upload or add a dataset first.")
    else:
        plot_cols = st.columns(2)

        # 1) Age distribution split by Survived
        if "Age" in df.columns and TARGET_COL in df.columns:
            fig = px.histogram(df, x="Age", color=TARGET_COL, nbins=30, barmode="overlay", title="Age Distribution by Survival")
            plot_cols[0].plotly_chart(fig, use_container_width=True)

        # 2) Survival rate by Sex
        if "Sex" in df.columns and TARGET_COL in df.columns:
            by_sex = df.groupby("Sex")[TARGET_COL].mean().reset_index()
            fig2 = px.bar(by_sex, x="Sex", y=TARGET_COL, title="Survival Rate by Sex")
            plot_cols[1].plotly_chart(fig2, use_container_width=True)

        # 3) Survival by Pclass
        if "Pclass" in df.columns and TARGET_COL in df.columns:
            by_pclass = df.groupby("Pclass")[TARGET_COL].mean().reset_index()
            fig3 = px.bar(by_pclass, x="Pclass", y=TARGET_COL, title="Survival Rate by Passenger Class")
            st.plotly_chart(fig3, use_container_width=True)

        # 4) Fare vs Age scatter colored by Survived
        if set(["Fare", "Age", TARGET_COL]).issubset(df.columns):
            fig4 = px.scatter(df, x="Age", y="Fare", color=TARGET_COL, hover_data=["Sex","Pclass","Embarked"], title="Fare vs Age (colored by Survival)")
            st.plotly_chart(fig4, use_container_width=True)

elif section == "Model & Prediction":
    st.header("Model & Prediction")
    if df is None:
        st.error("Please upload or add a dataset first.")
    else:
        if TARGET_COL not in df.columns:
            st.warning("Target column 'Survived' not found. Add it to train a model, or upload a file that includes it.")
        model, metrics = ensure_model(df)

        st.subheader("Enter Passenger Details")
        c1, c2, c3 = st.columns(3)
        sex = c1.selectbox("Sex", ["male", "female"])
        pclass = c2.selectbox("Passenger Class (Pclass)", [1, 2, 3], index=2)
        embarked = c3.selectbox("Embarked", ["S", "C", "Q"])

        c4, c5, c6, c7 = st.columns(4)
        age = c4.number_input("Age", min_value=0.0, max_value=100.0, value=29.0, step=1.0)
        sibsp = c5.number_input("Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0, step=1)
        parch = c6.number_input("Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0, step=1)
        fare = c7.number_input("Fare", min_value=0.0, max_value=600.0, value=32.2, step=0.1)

        X_input = pd.DataFrame([{
            "Sex": sex,
            "Pclass": pclass,
            "Embarked": embarked,
            "Age": age,
            "SibSp": sibsp,
            "Parch": parch,
            "Fare": fare
        }])

        if st.button("Predict Survival"):
            with st.spinner("Predicting..."):
                prob = probability_from_model(model, X_input)[0]
                pred = int(prob >= 0.5)
            st.success(f"Prediction: {'Survived (1)' if pred == 1 else 'Did NOT Survive (0)'}")
            st.info(f"Confidence (approx.): {prob:.3f}")

elif section == "Performance":
    st.header("Model Performance")
    if df is None:
        st.error("Please upload or add a dataset first.")
    else:
        model, metrics = ensure_model(df)
        if metrics is None:
            st.warning("No metrics available yet.")
        else:
            colA, colB = st.columns(2)
            cv_scores = metrics.get("cv_scores", {})
            best_model = metrics.get("best_model", "N/A")
            holdout_acc = metrics.get("holdout_accuracy", None)
            cm = np.array(metrics.get("confusion_matrix", []))

            with colA:
                st.subheader("Cross-Validation Accuracy")
                if cv_scores:
                    cv_df = pd.DataFrame({
                        "Model": list(cv_scores.keys()),
                        "Accuracy": list(cv_scores.values())
                    }).sort_values("Accuracy", ascending=False)
                    st.dataframe(cv_df, use_container_width=True)
                    fig = px.bar(cv_df, x="Model", y="Accuracy", title="Model Comparison (CV acc)")
                    st.plotly_chart(fig, use_container_width=True)
                st.metric("Best Model", best_model)

            with colB:
                if holdout_acc is not None:
                    st.metric("Holdout Accuracy", f"{holdout_acc:.3f}")
                if cm.size > 0:
                    st.subheader("Confusion Matrix (Holdout)")
                    fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual", color="Count"),
                                       x=[0,1], y=[0,1], title="Confusion Matrix")
                    st.plotly_chart(fig_cm, use_container_width=True)

            with st.expander("Classification Report (Holdout)"):
                rep = metrics.get("classification_report", {})
                rep_df = pd.DataFrame(rep).T
                st.dataframe(rep_df, use_container_width=True)


