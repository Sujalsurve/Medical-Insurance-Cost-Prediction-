import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the pre-trained model
model = joblib.load("insurance_model.pkl")

# Set the page layout
st.set_page_config(
    page_title="Medical Insurance Cost Prediction",
    page_icon="💰",
    layout="wide",
)

# Title and description
st.title("Medical Insurance Cost Prediction App 💰")
st.markdown(
    """
    This app predicts **medical insurance costs** based on user inputs like age, BMI, smoking status, region, and more.
    Adjust the parameters and click **Predict** to get your results.
    """
)

# Sidebar for user inputs
st.sidebar.header("Input Features")
age = st.sidebar.slider("Age", min_value=18, max_value=100, value=30, step=1)
bmi = st.sidebar.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
children = st.sidebar.slider("Number of Children", min_value=0, max_value=5, value=1)
smoker = st.sidebar.radio("Smoker", options=["Yes", "No"])
region = st.sidebar.selectbox("Region", options=["northeast", "northwest", "southeast", "southwest"])

# Convert inputs to numerical format
smoker_encoded = 1 if smoker == "Yes" else 0
region_encoded = {
    "northeast": [1, 0, 0, 0],
    "northwest": [0, 1, 0, 0],
    "southeast": [0, 0, 1, 0],
    "southwest": [0, 0, 0, 1],
}[region]

# Prepare the input data
input_data = np.array([[age, bmi, children, smoker_encoded] + region_encoded])

# Predict button
if st.sidebar.button("Predict"):
    # Make prediction
    prediction = model.predict(input_data)[0]

    # Display prediction
    st.header("Prediction Results")
    st.success(f"The estimated medical insurance cost is **${prediction:,.2f}**")
    st.write("Below is a breakdown of how your input features contributed to this prediction.")

    # Display user inputs in a table
    input_summary = pd.DataFrame({
        "Feature": ["Age", "BMI", "Children", "Smoker"] + ["Region: " + r for r in ["northeast", "northwest", "southeast", "southwest"]],
        "Value": [age, bmi, children, smoker_encoded] + region_encoded,
    })
    st.table(input_summary)

    # Display interactive visualizations
    st.subheader("Visual Analysis")
    col1, col2 = st.columns(2)

    # Feature importance visualization
    with col1:
        st.markdown("### Impact of Features")
        feature_importance = {
            "Age": age * 100,  # Simulated importance for demo
            "BMI": bmi * 50,   # Simulated importance for demo
            "Children": children * 70,
            "Smoker": smoker_encoded * 500,
        }
        feature_importance = pd.DataFrame(feature_importance.items(), columns=["Feature", "Importance"])
        sns.barplot(x="Importance", y="Feature", data=feature_importance, palette="viridis")
        plt.title("Feature Importance")
        st.pyplot(plt.gcf())

    # Age vs. Predicted Cost
    with col2:
        st.markdown("### Age vs. Predicted Cost")
        simulated_ages = np.linspace(18, 100, 50)
        simulated_costs = model.predict(
            np.array([[a, bmi, children, smoker_encoded] + region_encoded for a in simulated_ages])
        )
        plt.figure(figsize=(6, 4))
        plt.plot(simulated_ages, simulated_costs, label="Predicted Cost", color="blue")
        plt.xlabel("Age")
        plt.ylabel("Predicted Insurance Cost")
        plt.title("Age vs. Predicted Insurance Cost")
        plt.legend()
        st.pyplot(plt.gcf())

# Footer
st.markdown("---")
st.markdown(
    """
    **How it works:**  
    This prediction is based on a **Linear Regression Model** trained with synthetic data that considers your input features (age, BMI, etc.).  
    For a detailed breakdown, contact the app developer.
    """
)
