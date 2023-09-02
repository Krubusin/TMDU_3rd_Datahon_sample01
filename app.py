import streamlit as st
import xgboost as xgb
import numpy as np
import pickle

# モデルをファイルから読み込む
with open('xgboost_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# ページのタイトルを設定
st.title("SpO2_SaO2 Predictor")

# ユーザーからの数値入力を受け取る
sex_female = st.number_input("Sex (0 for male, 1 for female)", 0, 1, 0)
anchor_age = st.number_input("Anchor Age", min_value=0)
BMI = st.number_input("BMI")
mbp = st.number_input("MBP")
resp_rate = st.number_input("Respiratory Rate")
heart_rate = st.number_input("Heart Rate")
vasopress = st.number_input("Vasopress", 0, 1, 0)
hemoglobin = st.number_input("Hemoglobin")
SpO2 = st.number_input("SpO2")
selected_race = st.selectbox("Race", ["White", "Black", "Hispanic", "Asian"])

# 選択されたRaceを1、他のRaceを0にエンコード
race_encoded = {race: 1 if race == selected_race else 0 for race in ["White", "Black", "Hispanic", "Asian"]}

# ボタンを押して予測を実行
if st.button("Predict"):
    # 入力データをXGBoostモデルに適用して予測を行う
    input_data = np.array([[sex_female, anchor_age, BMI, mbp, resp_rate, heart_rate, vasopress, hemoglobin, SpO2] + list(race_encoded.values())])
    prediction_proba = loaded_model.predict_proba(input_data)[:, 1]

    # 予測結果を表示
    st.header("Predicted Probability (SpO2_SaO2 > 3)")
    st.header(f"**{prediction_proba[0] * 100:.2f}%**")




