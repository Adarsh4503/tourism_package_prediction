"""Streamlit front-end for the Wellness Tourism Package predictor."""
import os

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.joblib")

st.set_page_config(
    page_title="Wellness Tourism Package Predictor",
    page_icon="✈️",
    layout="wide",
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

st.title("Wellness Tourism Package Prediction")
st.write(
    "Predict whether a customer is likely to purchase the newly "
    "introduced Wellness Tourism Package before contacting them."
)

with st.form("input_form"):
    st.subheader("Customer Profile")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        typeofcontact = st.selectbox(
            "TypeofContact", ["Company Invited", "Self Inquiry"]
        )
        citytier = st.selectbox("CityTier", [1, 2, 3], index=0)
        occupation = st.selectbox(
            "Occupation",
            ["Salaried", "Freelancer", "Small Business", "Large Business"],
        )
        gender = st.selectbox("Gender", ["Male", "Female"])
        numberofpersonvisiting = st.number_input(
            "NumberOfPersonVisiting", min_value=1, max_value=10, value=2
        )

    with col2:
        preferredpropertystar = st.selectbox(
            "PreferredPropertyStar", [3, 4, 5], index=1
        )
        maritalstatus = st.selectbox(
            "MaritalStatus", ["Single", "Married", "Divorced"]
        )
        numberoftrips = st.number_input(
            "NumberOfTrips", min_value=0, max_value=20, value=3
        )
        passport = st.selectbox("Passport", [0, 1], index=1)
        owncar = st.selectbox("OwnCar", [0, 1], index=1)
        numberofchildrenvisiting = st.number_input(
            "NumberOfChildrenVisiting", min_value=0, max_value=5, value=0
        )

    with col3:
        designation = st.selectbox(
            "Designation",
            ["Executive", "Manager", "Senior Manager", "AVP", "VP"],
        )
        monthlyincome = st.number_input(
            "MonthlyIncome",
            min_value=1000,
            max_value=200000,
            value=30000,
            step=500,
        )
        pitchsatisfactionscore = st.selectbox(
            "PitchSatisfactionScore", [1, 2, 3, 4, 5], index=3
        )
        productpitched = st.selectbox(
            "ProductPitched",
            ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"],
        )
        numberoffollowups = st.number_input(
            "NumberOfFollowups", min_value=0, max_value=10, value=3
        )
        durationofpitch = st.number_input(
            "DurationOfPitch", min_value=1, max_value=120, value=15
        )

    submit = st.form_submit_button("Predict")

if submit:
    input_df = pd.DataFrame(
        [
            {
                "Age": age,
                "TypeofContact": typeofcontact,
                "CityTier": citytier,
                "Occupation": occupation,
                "Gender": gender,
                "NumberOfPersonVisiting": numberofpersonvisiting,
                "PreferredPropertyStar": preferredpropertystar,
                "MaritalStatus": maritalstatus,
                "NumberOfTrips": numberoftrips,
                "Passport": passport,
                "OwnCar": owncar,
                "NumberOfChildrenVisiting": numberofchildrenvisiting,
                "Designation": designation,
                "MonthlyIncome": monthlyincome,
                "PitchSatisfactionScore": pitchsatisfactionscore,
                "ProductPitched": productpitched,
                "NumberOfFollowups": numberoffollowups,
                "DurationOfPitch": durationofpitch,
            }
        ]
    )

    prediction = int(model.predict(input_df)[0])
    probability = None
    try:
        probability = float(model.predict_proba(input_df)[0][1])
    except Exception:
        probability = None

    st.subheader("Prediction")
    if prediction == 1:
        st.success(
            "The customer is **LIKELY** to purchase the Wellness "
            "Tourism Package."
        )
    else:
        st.warning(
            "The customer is **UNLIKELY** to purchase the Wellness "
            "Tourism Package."
        )

    if probability is not None:
        st.metric("Purchase probability", f"{probability:.2%}")

    with st.expander("Submitted feature values"):
        st.dataframe(input_df.T, use_container_width=True)
