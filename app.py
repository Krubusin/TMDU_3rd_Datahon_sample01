import streamlit as st
import xgboost as xgb
import numpy as np
import pickle
# モデルをファイルから読み込む
with open('xgboost_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# ページのタイトルを設定
st.title("The need to take ABG")

# ユーザーからの数値入力を受け取る
SpO2 = st.slider("SpO2",min_value=90.0, max_value=100.0, value=98.0)
SpO2 = int(SpO2)
RR = st.slider("Respiratory Rate", min_value=10.0, max_value=70.0, value=20.0, step=0.1)
MBP = st.slider("MBP", min_value=60.0, max_value=200.0, value=75.0, step=0.1)
HR = st.slider("Heart Rate", min_value=23.0, max_value=150.0, value=85.0, step=0.1)
# vasopress = st.number_input("Vasopress Use", 0, 1, 0)
vasopress_0 = st.radio(
    "Vasopressor Use?",
    ["Yes", "No"])
Vasopressor = (0 if vasopress_0 == 'No' else 1)

Ventilation_0 = st.radio(
    "Invasive_ventilation Use?",
    ["Yes", "No"])
Ventilation = (0 if Ventilation_0 == 'No' else 1)
Temperature = st.slider("Temperature",min_value=34, max_value=44, value=36)
sex = st.radio(
    "Your patient is?",
    ["Male", "Female"])
Female = (0 if sex == 'Male' else 1)
BMI = st.slider("BMI", min_value=15.0, max_value=45.0, value=28.0, step=0.1)
Age = st.slider('How old are your patient?', 0, 100, 65)
Age = int(Age)
# st.write("He/She is ", anchor_age, 'years old')

# selected_race = st.selectbox("Race", ["White", "Black", "Hispanic", "Asian"])
selected_race = st.radio(
    "Which Race is your patient?",
    ["Black", "White", "Hispanic", "Asian"])



# 選択されたRaceを1、他のRaceを0にエンコード
race_encoded = {race: 1 if race == selected_race else 0 for race in ["Black", "White", "Hispanic", "Asian"]}

# ボタンを押す代わりに入力データで予測を実行
input_data = np.array([[Female, Age, MBP, RR, HR, Vasopressor, SpO2, BMI,  Ventilation, Temperature] + list(race_encoded.values())])
prediction_proba = loaded_model.predict_proba(input_data)[:, 1]

# 予測結果を表示
st.header("Predicted Probability (SpO2-SaO2 >= 3)")
st.header(f"**{prediction_proba[0] * 100:.2f}%**")
