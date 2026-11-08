import streamlit as st
import joblib
from pathlib import Path
from train_model import ensure_model

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

MODEL_PATH = Path(__file__).parent / "titanic_model.joblib"

@st.cache_resource
def load_model():
    ensure_model()
    return joblib.load(MODEL_PATH)

st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details below to estimate the model's survival prediction.")
st.info("This is an educational machine-learning demo. Predictions are estimates, not historical certainty.")

model = load_model()

col1, col2 = st.columns(2)
with col1:
    pclass = st.selectbox("Passenger class", [1, 2, 3], index=2)
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.slider("Age", 0, 80, 25)
    sibsp = st.slider("Siblings / spouses aboard", 0, 8, 0)
with col2:
    parch = st.slider("Parents / children aboard", 0, 6, 0)
    fare = st.number_input("Ticket fare", min_value=0.0, max_value=600.0, value=32.0, step=1.0)
    embarked = st.selectbox("Port of embarkation", ["S", "C", "Q"], format_func=lambda x: {"S":"Southampton", "C":"Cherbourg", "Q":"Queenstown"}[x])

if st.button("Predict survival", type="primary", use_container_width=True):
    import pandas as pd
    passenger = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked
    }])
    prediction = int(model.predict(passenger)[0])
    probability = float(model.predict_proba(passenger)[0][1])
    if prediction == 1:
        st.success(f"Prediction: Likely to survive ({probability:.1%} estimated probability)")
    else:
        st.error(f"Prediction: Not likely to survive ({probability:.1%} estimated survival probability)")
    st.progress(probability)
    st.caption("Model: Decision Tree Classifier trained on the Titanic dataset.")
