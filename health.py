"Used chat GPT for debugging styling and coloring of text in the output dataframe"

# Import necessary libraries
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('fetal_health.csv')


st.title("Fetal Health Classification: A Machine Learning App")
# Display an image of penguins
st.image('fetal_health_image.gif', width = 600)
st.write("Utilize our advanced Machine Learning application to predict fetal health classification")


st.sidebar.header('**Featal Health Feature Importance**')

#upload file
uploaded_file = st.sidebar.file_uploader("Upload your data", type="csv", help="Upload a CSV file containing fetal health data.")

st.sidebar.warning("⚠️ Ensure your data strictly follows the format outlined below")
st.sidebar.dataframe(df.head())

model = st.sidebar.radio("Choose Model for Prediction", ('Random Forest','Decision Tree', 'AdaBoost', 'Soft Voting'))

st.sidebar.info(f" ✓ You Selected: {model}")

if uploaded_file is None:
    st.info(" ℹ️Please upload data to proceed")

if uploaded_file is not None:
    if model == 'Random Forest':
        loaded_model = pickle.load(open('health_rf.pickle', 'rb'))
        feature_importance ="feature_imp_health_rf.svg"
        confusion_matrix= "confusion_mat_rf_health.svg"
        class_report = "class_report_health_rf.csv"
        colors="Greens"
    elif model == 'Decision Tree':
        loaded_model = pickle.load(open('health_dt.pickle', 'rb'))
        feature_importance ="feature_imp_health_dt.svg"
        confusion_matrix= "confusion_mat__dt_health.svg"
        class_report = "class_report_health_dt.csv"
        colors="Reds"
    elif model == 'AdaBoost':
        loaded_model = pickle.load(open('health_ada.pickle', 'rb'))  
        feature_importance ="feature_imp_health_ada.svg"
        confusion_matrix= "confusion_mat__ada_health.svg"
        class_report = "class_report_health_ada.csv"
        colors="Blues"
    elif model == 'Soft Voting':
        loaded_model = pickle.load(open('health_vote.pickle', 'rb'))
        feature_importance ="feature_imp_health_vote.svg"
        confusion_matrix= "confusion_mat__vote_health.svg"
        class_report = "class_report_health_vote.csv"
        colors="Oranges"

    data = pd.read_csv(uploaded_file)

    st.success("✅ CSV file uploaded sucessfully")
    st.subheader(f"Predicting Fetal Health Class Using {model}")

    # Predict class and probabilities
    predictions = loaded_model.predict(data)
    prediction_probs = loaded_model.predict_proba(data)
    classes = loaded_model.classes_ 

    data['Predicted Fetal Health'] = predictions
    predicted_probs = []
    for i, pred in enumerate(predictions):
        # Find index of the predicted class in model.classes_
        class_index = np.where(classes == pred)[0][0]
        predicted_probs.append(prediction_probs[i, class_index]*100)

    data['Prediction Probability'] = np.round(predicted_probs, 1)
    data['Prediction Probability'] = data['Prediction Probability'].map(lambda x: f"{x:.1f}")

    # Define color function
    def color_prediction(val):
        if val == 'Normal':
            color = "#27f10c"  # light green
        elif val == 'Suspect':
            color = "#f8f805"  # light yellow
        elif val == 'Pathological':
            color = "#fc8e08"  # light orange
        else:
            color = 'white'
        return f'background-color: {color}; font-weight: bold;'

    # Apply styling
    styled_df = data.style.applymap(color_prediction, subset=['Predicted Fetal Health'])

    # Show in Streamlit
    st.dataframe(styled_df)


    st.subheader("Model Preformance and Insight")
    tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Classification Report",  "Feature Importance"])
    with tab1:
        st.write("### Confusion Matrix")
        st.image(confusion_matrix)
        st.caption("Confusion matrix for the model.")

    with tab2:
        st.write("### Classification Report")
        report_df = pd.read_csv(class_report, index_col = 0).transpose()
        st.dataframe(report_df.style.background_gradient(cmap=colors).format(precision=2))
        st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each health condition.")



    with tab3:
        st.write("### Feature Importance")
        st.image(feature_importance)
        st.caption("Features used in this prediction are ranked by relative importance.")
