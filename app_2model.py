import streamlit as st
import xgboost as xgb
import numpy as np
import pickle
# モデルをファイルから読み込む
with open('xgboost_model_allrace.pkl', 'rb') as f:
    loaded_model_allrace = pickle.load(f)

with open('xgboost_model_black.pkl', 'rb') as t:
    loaded_model_black = pickle.load(t)

# ページのタイトルを設定
st.title("SpO2-SaO2 Predictor")

# ユーザーからの数値入力を受け取る
# sex_female = st.number_input("Sex (0 for male, 1 for female)", 0, 1, 0)
selected_race = st.radio(
    "Which Race is your patient?",
    ["White", "Black", "Hispanic","Asian"])
race_encoded = {race: 1 if race == selected_race else 0 for race in ["White", "Black", "Hispanic", "Asian"]}
choice = st.radio(
    "You use only Black Race Trained Model?",
    ["Yes", "No"])

sex = st.radio(
    "Your patient is?",
    ["Male", "Female"])
sex_female = (0 if sex == 'Male' else 1)
# anchor_age = st.number_input("Anchor Age", min_value=0)
anchor_age = st.slider('How old are your patient?', 0, 120, 65)
# st.write("He/She is ", anchor_age, 'years old')
BMI = st.slider("BMI", min_value=15.0, max_value=60.0, value=28.0, step=0.1)
mbp = st.slider("MBP", min_value=60.0, max_value=300.0, value=75.0, step=0.1)
resp_rate = st.slider("Respiratory Rate", min_value=10.0, max_value=70.0, value=20.0, step=0.1)
heart_rate = st.slider("Heart Rate", min_value=23.0, max_value=300.0, value=85.0, step=0.1)
# vasopress = st.number_input("Vasopress Use", 0, 1, 0)
vasopress_0 = st.radio(
    "Vasopress Use?",
    ["Yes", "No"])
vasopress = (0 if vasopress_0 == 'No' else 1)
SpO2 = st.slider("SpO2",min_value=80.0, max_value=300.0, value=98.0, step=0.1)
# selected_race = st.selectbox("Race", ["White", "Black", "Hispanic", "Asian"])
vent_0 = st.radio(
    "invasive_vent Use?",
    ["Yes", "No"])
vent = (0 if vent_0 == 'No' else 1)
temperature = st.slider("temperature",min_value=34.0, max_value=44.0, value=36.0, step=0.1)


if choice == "Yes":

    Black = 1 if selected_race == 'Black' else 0
    input_data_black = np.array([[sex_female, anchor_age, BMI, mbp, resp_rate, heart_rate, vasopress, vent,temperature, SpO2, Black]])
    prediction_proba_black = loaded_model_black.predict_proba(input_data_black)[:, 1]
    st.header("Predicted Probability (SpO2-SaO2 >= 3)")
    st.header(f"**{prediction_proba_black[0] * 100:.2f}%**")

else:
    race_encoded_all = {race: 1 if race == selected_race else 0 for race in ["White", "Black", "Hispanic", "Asian"]}
    input_data_all = np.array([[sex_female, anchor_age, BMI, mbp, resp_rate, heart_rate, vasopress, vent,temperature, SpO2] + list(race_encoded.values())])
    prediction_proba_all = loaded_model_allrace.predict_proba(input_data_all)[:, 1]
    st.header("Predicted Probability (SpO2-SaO2 >= 3)")
    st.header(f"**{prediction_proba_all[0] * 100:.2f}%**")
# 選択されたRaceを1、他のRaceを0にエンコード
#race_encoded = {race: 1 if race == selected_race else 0 for race in ["White", "Black", "Hispanic", "Asian"]}


# ボタンを押す代わりに入力データで予測を実行
# input_data_all = np.array([[sex_female, anchor_age, BMI, mbp, resp_rate, heart_rate, vasopress, vent,temperature, SpO2] + list(race_encoded.values())])
# input_data_black = np.array([[sex_female, anchor_age, BMI, mbp, resp_rate, heart_rate, vasopress, vent,temperature, SpO2]])
# prediction_proba_all = loaded_model_allrace.predict_proba(input_data_all)[:, 1]
# prediction_proba_black = loaded_model_black.predict_proba(input_data_black)[:, 1]
# 予測結果を表示
# st.header("Predicted Probability (SpO2_SaO2 >= 3)")
# st.header(f"**{prediction_proba_all[0] * 100:.2f}%**")
# st.header(f"**{prediction_proba_black[0] * 100:.2f}%**")
