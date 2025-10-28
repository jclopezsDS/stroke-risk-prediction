import pandas as pd
import shap
import plotly.express as px

def preprocess_input(input_data):
    df = input_data.copy()
    df['hypertension'] = df['hypertension'].map({'Yes': 1, 'No': 0})
    df['heart_disease'] = df['heart_disease'].map({'Yes': 1, 'No': 0})
    df['residence_urban'] = df['residence_type'].map({'Urban': 1, 'Rural': 0})
    df['gender_male'] = df['gender'].map({'Male': 1, 'Female': 0})
    df['ever_married'] = df['ever_married'].map({'Yes': 1, 'No': 0})
    
    work_type_dummies = pd.get_dummies(df['work_type'], prefix='work_type').reindex(
        columns=['work_type_govt_job', 'work_type_never_worked', 'work_type_private',
                 'work_type_self-employed', 'work_type_children'], fill_value=0).astype(int)
    df = pd.concat([df, work_type_dummies], axis=1).drop(['residence_type', 'gender', 'work_type'], axis=1)
    
    smoking_dummies = pd.get_dummies(df['smoking_status'], prefix='smoking_status').reindex(
        columns=['smoking_status_never smoked', 'smoking_status_formerly smoked', 'smoking_status_smokes'],
        fill_value=0).astype(int)
    df = pd.concat([df, smoking_dummies], axis=1).drop(['smoking_status'], axis=1)
    
    df['age_glucose_interaction'] = df['age'] * df['avg_glucose_level']
    df['age_hypertension_interaction'] = df['age'] * df['hypertension']
    df['bmi_glucose_interaction'] = df['bmi'] * df['avg_glucose_level']
    df['smoking_age_interaction'] = df['age'] * df['smoking_status_smokes']
    df['age_squared'] = df['age'] ** 2
    df['glucose_squared'] = df['avg_glucose_level'] ** 2
    df['bmi_squared'] = df['bmi'] ** 2
    df['age_under_45'] = (df['age'] < 45).astype(int)
    df['age_over_65'] = (df['age'] > 65).astype(int)
    df['glucose_over_200'] = (df['avg_glucose_level'] > 200).astype(int)
    df['bmi_under_25'] = (df['bmi'] < 25).astype(int)
    
    feature_cols = ['age', 'hypertension', 'heart_disease', 'ever_married', 'avg_glucose_level', 'bmi',
                    'residence_urban', 'gender_male', 'work_type_govt_job', 'work_type_never_worked',
                    'work_type_private', 'work_type_self-employed', 'work_type_children',
                    'smoking_status_never smoked', 'smoking_status_formerly smoked', 'smoking_status_smokes',
                    'age_under_45', 'age_squared', 'age_over_65', 'age_glucose_interaction',
                    'age_hypertension_interaction', 'smoking_age_interaction', 'bmi_glucose_interaction',
                    'glucose_squared', 'bmi_squared']
    df = df[feature_cols]
    
    df = df.astype(float)
    return df

def generate_shap_plot(model, processed_input):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(processed_input.to_numpy())
    feature_names = processed_input.columns.tolist()
    shap_df = pd.DataFrame({
        'feature': feature_names,
        'shap_value': shap_values[0]
    }).sort_values('shap_value', key=abs, ascending=False).head(5)
    
    fig = px.bar(
        shap_df,
        x='shap_value',
        y='feature',
        orientation='h',
        title='Top 5 Factors Influencing Stroke Risk Prediction',
        labels={'shap_value': 'Impact on Prediction', 'feature': 'Feature'},
        color='shap_value',
        color_continuous_scale='RdBu_r',
        range_color=[-max(abs(shap_df['shap_value'])), max(abs(shap_df['shap_value']))]
    )
    fig.update_layout(yaxis={'autorange': 'reversed'}, showlegend=False)
    return fig