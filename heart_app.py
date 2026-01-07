import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_PATH = "dataset/heart_disease.csv"
MODEL_PATH = "model/heart_model.keras"
SCALER_PATH = "model/scaler.pkl"
COLS_PATH = "model/feature_columns.json"

os.makedirs("model", exist_ok=True)

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

def preprocess(df):
    df2 = df.copy()
    for col in ["Cholesterol", "RestingBP"]:
        if col in df2.columns:
            df2[col] = df2[col].replace(0, np.nan)
            df2[col] = df2[col].fillna(df2[col].median())

    y = df2["HeartDisease"].astype(int)
    X = df2.drop(columns=["HeartDisease"])
    X = pd.get_dummies(X, drop_first=True)
    return X, y

def build_model(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.30),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.20),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")]
    )
    return model

def train_and_save(df):
    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = build_model(X_train_s.shape[1])

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", patience=7, factor=0.5),
    ]

    model.fit(
        X_train_s, y_train,
        validation_split=0.2,
        epochs=200,
        batch_size=32,
        callbacks=callbacks,
        verbose=0
    )

    proba = model.predict(X_test_s).ravel()
    pred = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, proba)
    cm = confusion_matrix(y_test, pred)
    report = classification_report(y_test, pred, digits=4)

    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(COLS_PATH, "w", encoding="utf-8") as f:
        json.dump(list(X.columns), f, ensure_ascii=False)

    return auc, cm, report

def load_model_and_scaler():
    model = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(COLS_PATH, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)
    return model, scaler, feature_cols

def make_input_df(user_dict, feature_cols):
    base = pd.DataFrame([user_dict])
    base = pd.get_dummies(base, drop_first=True)
    for c in feature_cols:
        if c not in base.columns:
            base[c] = 0
    return base[feature_cols]

st.set_page_config(page_title="HeartDisease App", layout="wide")
st.title("심장질환 여부 예측")

df = load_data(DATA_PATH)

left, right = st.columns(2)

with left:
    st.subheader("예측 전에 학습해주세요!")
    if st.button("모델 학습"):
        with st.spinner("Training..."):
            auc, cm, report = train_and_save(df)
        st.success("Done")
        st.write(f"ROC-AUC: {auc:.4f}")
        st.write("Confusion Matrix")
        st.write(cm)
        st.text(report)

with right:
    st.subheader("예측하기")

    age = st.number_input("나이", value=50, min_value=1)
    restingbp = st.number_input("안정된 상태에서의 혈압", value=120, min_value=0)
    cholesterol = st.number_input("콜레스트롤", value=200, min_value=0)
    fastingbs_text = st.radio("공복혈당", ["120 mg/dL 이하", "120 mg/dL 초과"], horizontal=True)
    fastingbs = 1 if fastingbs_text == "120 mg/dL 초과" else 0

    maxhr = st.number_input("최대 심박수", value=150, min_value=0)
    oldpeak = st.number_input("Oldpeak", value=1.0)

    if st.button("예측"):
        if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(COLS_PATH)):
            st.error("먼저 학습하세요.")
        else:
            model, scaler, feature_cols = load_model_and_scaler()
            user = {
                "Age": age,
                "RestingBP": restingbp,
                "Cholesterol": cholesterol,
                "FastingBS": fastingbs,
                "MaxHR": maxhr,
                "Oldpeak": oldpeak,
            }
            X_in = make_input_df(user, feature_cols)
            X_in_s = scaler.transform(X_in)

            proba = float(model.predict(X_in_s).ravel()[0])
            pred = 1 if proba >= 0.5 else 0

            st.write(f"질환 확률: {proba:.4f}")

            result_text = "심장질환이 있을 수 있습니다" if pred == 1 else "심장질환이 있을 확률이 낮습니다. "
            st.write(f"예측 결과: {result_text}")
            
            

