import streamlit as st
import pandas as pd
import joblib
import shap
from utils import preprocess_input, generate_shap_plot

model = joblib.load('xgb_model.pkl')

st.title("Stroke Risk Prediction")
st.write("Enter patient details to predict stroke risk. All fields are required.")

inputs = {
    'age': st.number_input("Age", min_value=0.0, max_value=120.0, value=30.0, step=1.0,
                          help="Age in years (0-120)"),
    'hypertension': st.selectbox("Hypertension", ['No', 'Yes']),
    'heart_disease': st.selectbox("Heart Disease", ['No', 'Yes']),
    'ever_married': st.selectbox("Ever Married", ['Yes', 'No']),
    'avg_glucose_level': st.number_input("Average Glucose Level", min_value=50.0, max_value=300.0,
                                       value=100.0, step=1.0, help="Glucose level in mg/dL (50-300)"),
    'bmi': st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1,
                          help="Body Mass Index (10-60)"),
    'residence_type': st.selectbox("Residence Type", ['Urban', 'Rural']),
    'gender': st.selectbox("Gender", ['Male', 'Female']),
    'work_type': st.selectbox("Work Type", ['Private', 'Self-employed']),
    'smoking_status': st.selectbox("Smoking Status", ['never smoked', 'formerly smoked', 'smokes'])
}

input_df = pd.DataFrame([inputs])

if st.button("Predict Stroke Risk"):
    if not all(input_df[col].notna().all() for col in inputs.keys()):
        st.error("Please fill all fields.")
    else:
        processed_input = preprocess_input(input_df)
        processed_input_array = processed_input.to_numpy()
        prob = model.predict_proba(processed_input_array)[:, 1][0]
        risk = "High" if prob > 0.5 else "Low"
        st.write(f"**Predicted Stroke Risk**: {risk} (Probability: {prob:.3f})")
        shap_fig = generate_shap_plot(model, processed_input)
        st.plotly_chart(shap_fig, use_container_width=True)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(processed_input_array)
        feature_names = processed_input.columns.tolist()
        shap_df = pd.DataFrame({
            'feature': feature_names,
            'shap_value': shap_values[0]
        }).sort_values('shap_value', key=abs, ascending=False).head(5)
        top_features = shap_df['feature'].tolist()
        top_values = [float(x) for x in shap_df['shap_value'].tolist()]
        input_values = [float(processed_input[feat].iloc[0]) for feat in top_features]

        st.write("**How to Read the Chart**:")
        st.write(f"This chart shows the top 5 factors affecting your stroke risk. Positive bars (red) mean the factor increases your risk, while negative bars (blue) lower it. Here’s what your values mean:")
        for feat, val, impact in zip(top_features, input_values, top_values):
            direction = "raises" if impact > 0 else "lowers"
            st.write(f"- {feat.replace('_', ' ').title()}: Your value is {val:.1f}, which {direction} your risk by {abs(impact):.3f}.")

st.write("**Note**: This app uses a trained XGBoost model with 19 key features, optimized for stroke prediction.")