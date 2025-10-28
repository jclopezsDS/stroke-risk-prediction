import streamlit as st
import pandas as pd
import joblib
import os
from utils import preprocess_input

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'xgb_model.pkl')

model = joblib.load(MODEL_PATH)

st.title("🏥 Stroke Risk Prediction System")
st.write("Enter patient details to predict stroke risk. All fields are required.")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Demographics")
    age = st.number_input("Age", min_value=0.0, max_value=120.0, value=50.0, step=1.0,
                          help="Age in years (0-120)")
    gender = st.selectbox("Gender", ['Male', 'Female'])
    ever_married = st.selectbox("Ever Married", ['Yes', 'No'])
    residence_type = st.selectbox("Residence Type", ['Urban', 'Rural'])
    work_type = st.selectbox("Work Type", ['Private', 'Self-employed', 'Govt job', 'Never worked', 'Children'])

with col2:
    st.subheader("Health Indicators")
    hypertension = st.selectbox("Hypertension", ['No', 'Yes'])
    heart_disease = st.selectbox("Heart Disease", ['No', 'Yes'])
    avg_glucose_level = st.number_input("Average Glucose Level (mg/dL)", 
                                       min_value=50.0, max_value=300.0,
                                       value=100.0, step=1.0, 
                                       help="Glucose level in mg/dL (50-300)")
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1,
                          help="Body Mass Index (10-60)")
    smoking_status = st.selectbox("Smoking Status", ['never smoked', 'formerly smoked', 'smokes', 'Unknown'])

st.markdown("---")

inputs = {
    'age': age,
    'hypertension': hypertension,
    'heart_disease': heart_disease,
    'ever_married': ever_married,
    'avg_glucose_level': avg_glucose_level,
    'bmi': bmi,
    'residence_type': residence_type,
    'gender': gender,
    'work_type': work_type,
    'smoking_status': smoking_status
}

input_df = pd.DataFrame([inputs])

if st.button("🔍 Predict Stroke Risk", type="primary", use_container_width=True):
    if not all(input_df[col].notna().all() for col in inputs.keys()):
        st.error("❌ Please fill all fields.")
    else:
        with st.spinner("Analyzing patient data..."):
            try:
                processed_input = preprocess_input(input_df)
                processed_input_array = processed_input.to_numpy()
                prob = model.predict_proba(processed_input_array)[:, 1][0]
                
                st.success("✅ Analysis Complete!")
                
                # Display results with color coding
                risk = "High" if prob > 0.5 else "Low"
                risk_color = "🔴" if prob > 0.5 else "🟢"
                
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric("Risk Level", f"{risk_color} {risk}")
                with col_b:
                    st.metric("Probability", f"{prob:.1%}")
                with col_c:
                    confidence = "High" if abs(prob - 0.5) > 0.3 else "Medium" if abs(prob - 0.5) > 0.15 else "Low"
                    st.metric("Confidence", confidence)
                
                # Risk interpretation
                st.markdown("---")
                st.subheader("📊 Risk Interpretation")
                
                if prob < 0.2:
                    st.info("✅ **Low Risk**: Patient shows minimal stroke risk indicators. Continue regular health monitoring.")
                elif prob < 0.4:
                    st.info("⚠️ **Moderate-Low Risk**: Some risk factors present. Recommend lifestyle modifications and regular check-ups.")
                elif prob < 0.6:
                    st.warning("⚠️ **Moderate-High Risk**: Multiple risk factors detected. Consider preventive interventions and closer monitoring.")
                else:
                    st.error("🚨 **High Risk**: Significant stroke risk detected. Immediate medical consultation and intervention recommended.")
                
                # Key factors (simplified without SHAP)
                st.markdown("---")
                st.subheader("🔑 Key Risk Factors")
                st.write("Based on clinical research, the most important factors for stroke risk are:")
                
                factors_text = []
                if age > 65:
                    factors_text.append(f"• **Age**: {age:.0f} years (increased risk after 65)")
                if hypertension == 'Yes':
                    factors_text.append("• **Hypertension**: Present (major risk factor)")
                if heart_disease == 'Yes':
                    factors_text.append("• **Heart Disease**: Present (significant risk factor)")
                if avg_glucose_level > 140:
                    factors_text.append(f"• **Glucose Level**: {avg_glucose_level:.0f} mg/dL (elevated)")
                if bmi > 30:
                    factors_text.append(f"• **BMI**: {bmi:.1f} (obese category)")
                if smoking_status == 'smokes':
                    factors_text.append("• **Smoking**: Current smoker (major risk factor)")
                    
                if factors_text:
                    for factor in factors_text:
                        st.markdown(factor)
                else:
                    st.success("✅ No major risk factors detected in this assessment.")
                    
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
                st.info("Please check that all input values are valid.")

st.markdown("---")
st.caption("**Note**: This tool uses a trained XGBoost model optimized for stroke prediction. Results should be interpreted by qualified healthcare professionals.")
st.caption("**Model Performance**: ROC-AUC 0.84-0.85 | Sensitivity 82% | Specificity 90%")

