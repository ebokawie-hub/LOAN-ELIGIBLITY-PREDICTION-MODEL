"""Loan Eligibility Predictor - Streamlit interface for the Random Forest model.

Run with:  streamlit run app.py
Needs 'loan_eligibility_model.pkl' in the same folder as this file.
"""
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Loan Eligibility Predictor", page_icon="🏦", layout="centered")

MODEL_PATH = Path(__file__).parent / "loan_eligibility_model.pkl"

# ---------------------------------------------------------------------------
# ENCODINGS - these MUST match the mappings used in your notebook.
# If your notebook used different values, change them here.
# ---------------------------------------------------------------------------
GENDER = {"Male": 1, "Female": 0}
YES_NO = {"Yes": 1, "No": 0}
EDUCATION = {"Graduate": 1, "Not Graduate": 0}
CREDIT_HISTORY = {"Good (meets guidelines)": 1, "Bad (does not meet guidelines)": 0}
AREAS = ["Urban", "Semiurban", "Rural"]  # Rural is the dropped (all-zero) choice


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def build_row(values: dict) -> pd.DataFrame:
    """Turn the form answers into the same numeric columns the model was trained on."""
    return pd.DataFrame([{
        "Gender": GENDER[values["gender"]],
        "Married": YES_NO[values["married"]],
        "Dependents": values["dependents"],
        "Education": EDUCATION[values["education"]],
        "Self_Employed": YES_NO[values["self_employed"]],
        "Applicant_Income": values["applicant_income"],
        "Coapplicant_Income": values["coapplicant_income"],
        "Loan_Amount": values["loan_amount"],
        "Loan_Amount_Term": values["loan_term"],
        "Credit_History": CREDIT_HISTORY[values["credit_history"]],
        # One-hot encoding with drop_first=True: Rural = both switches off
        "Property_Area_Semiurban": int(values["area"] == "Semiurban"),
        "Property_Area_Urban": int(values["area"] == "Urban"),
    }])


# ------------------------------- Page header -------------------------------
st.title("🏦 Loan Eligibility Predictor")
st.write("Fill in the applicant's details. A Random Forest of 100 decision trees "
         "will vote on whether the loan is likely to be approved.")

if not MODEL_PATH.exists():
    st.error(f"Model file not found. Put `{MODEL_PATH.name}` in the same folder as app.py.")
    st.stop()

model = load_model()

with st.sidebar:
    st.header("About this model")
    st.write("**Algorithm:** Random Forest Classifier")
    st.write(f"**Trees:** {len(model.estimators_)}")
    st.write("**Test accuracy:** 81.3% (100 of 123 test customers)")
    st.caption("Demo only. This is not a real lending decision.")

# --------------------------------- The form --------------------------------
with st.form("loan_form"):
    c1, c2 = st.columns(2)
    with c1:
        gender = st.selectbox("Gender", list(GENDER))
        married = st.selectbox("Married", list(YES_NO))
        dependents = st.number_input("Number of dependents", min_value=0, max_value=10, value=0, step=1)
        education = st.selectbox("Education", list(EDUCATION))
        self_employed = st.selectbox("Self employed", list(YES_NO), index=1)
        area = st.selectbox("Property area", AREAS)
    with c2:
        applicant_income = st.number_input("Applicant income", min_value=0.0, value=5000.0, step=100.0)
        coapplicant_income = st.number_input("Co-applicant income", min_value=0.0, value=0.0, step=100.0)
        loan_amount = st.number_input("Loan amount (same units as your dataset)", min_value=0.0, value=130.0, step=1.0)
        loan_term = st.selectbox("Loan term (months)", [12, 36, 60, 84, 120, 180, 240, 300, 360, 480], index=8)
        credit_history = st.selectbox("Credit history", list(CREDIT_HISTORY))
    submitted = st.form_submit_button("Check eligibility", use_container_width=True)

# ------------------------------- Prediction --------------------------------
if submitted:
    row = build_row({
        "gender": gender, "married": married, "dependents": dependents,
        "education": education, "self_employed": self_employed, "area": area,
        "applicant_income": applicant_income, "coapplicant_income": coapplicant_income,
        "loan_amount": loan_amount, "loan_term": loan_term, "credit_history": credit_history,
    })

    # Make sure the columns match the model exactly, in the same order.
    expected = list(getattr(model, "feature_names_in_", row.columns))
    missing = [c for c in expected if c not in row.columns]
    extra = [c for c in row.columns if c not in expected]
    if missing or extra:
        st.error("The form's columns do not match the columns your model was trained on.")
        st.write("Model expects:", expected)
        st.write("Missing from form:", missing, "| Not used by model:", extra)
        st.info("Fix the column names in `build_row()` so they match your notebook.")
        st.stop()
    row = row[expected]

    # The target may be stored as 1/0 or as the letters Y/N, so find the "approved" label.
    classes = list(model.classes_)
    approved_class = next((c for c in ("Y", "Yes", "Approved", 1, "1") if c in classes), None)
    if approved_class is None:
        st.error(f"Could not tell which class means 'approved'. The model's classes are: {classes}")
        st.stop()
    probability = float(model.predict_proba(row)[0][classes.index(approved_class)])
    arr = row.to_numpy()
    votes = sum(int(model.classes_[int(t.predict(arr)[0])] == approved_class) for t in model.estimators_)
    total = len(model.estimators_)

    st.divider()
    if probability >= 0.5:
        st.success(f"✅ Likely to be APPROVED ({probability:.0%} confidence)")
    else:
        st.error(f"❌ Likely to be DENIED ({1 - probability:.0%} confidence)")
    st.progress(probability, text=f"{votes} of {total} trees voted to approve")

    with st.expander("What the model looked at (the numbers it received)"):
        st.dataframe(row.T.rename(columns={0: "Value"}))

with st.expander("Which facts matter most to the model?"):
    importances = pd.Series(model.feature_importances_,
                            index=getattr(model, "feature_names_in_", range(model.n_features_in_)))
    st.bar_chart(importances.sort_values(ascending=False))
