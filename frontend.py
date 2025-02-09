import streamlit as st
import requests
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px

# API URL
API_URL = "https://customer-churn-prediction-yn10.onrender.com/predict"

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔍",
    layout="wide",
)

# Custom CSS Styling
st.markdown("""
    <style>
    /* Main title styling */
    h1 {
        color: #4F8BF9;
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 20px;
    }

    /* Subheader styling */
    h2 {
        color: #2E86C1;
        font-size: 1.5em;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    
    /* Radio button and slider styling */
    .stRadio > div {
        flex-direction: column;
        align-items: flex-start;
    }

    .stRadio label {
        font-size: 1.1em;
        margin-bottom: 5px;
    }

    .stSlider {
        margin-top: 10px;
    }

    /* Button styling */            
    .stButton>button {
        background-color: #4F8BF9;
        color: white;
        font-size: 1.2em;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        width: 100%;
        margin-top: 20px;
    }
    .stButton button:hover {
        background-color: #2E86C1;
    }
    
    /* Success and error message styling */
    .stSuccess {
        font-size: 1.2em;
        color: #28B463;
        text-align: center;
        margin-top: 20px;
    }

    .stError {
        font-size: 1.2em;
        color: #E74C3C;
        text-align: center;
        margin-top: 20px;
    }

    /* Column spacing */
    .stColumn {
        padding: 10px;
    }

    /* Hyperlink styling */
    .predict-link {
        color: #4F8BF9;
        text-decoration: underline;
        cursor: pointer;
    }
    
    .stNumberInput input {
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("📊 Customer Churn Predictor")
st.markdown("""
    This application predicts whether a customer is likely to churn based on their details.
    Fill in the form below and click **<span class="predict-link" onclick="document.getElementById('predict-button').click()">Predict Churn</span>** to see the result.
""", unsafe_allow_html=True)
st.markdown("---")

# Divide the form into columns for better layout
col1, col2, col3, col4, col5 = st.columns([3, 3, 3, 3, 5])

# Input fields for customer data
with col1:
    st.subheader("Personal Details")
    gender = st.radio("Gender", ("Male", "Female"), help="Choose the gender of the customer (Male/Female)")
    senior_citizen = st.radio("Senior Citizen", ("Yes", "No"), help="Is the customer a senior citizen (Yes/No)")
    partner = st.radio("Partner", ("Yes", "No"), help="Does the customer have a partner? (Yes/No)")
    dependents = st.radio("Dependents", ("Yes", "No"), help="Does the customer have dependents? (Yes/No)")

with col2:
    st.subheader("Phone & Internet")
    phone_service = st.radio("Phone Service", ("Yes", "No"), help="Does the customer have phone service? (Yes/No)")
    multiple_lines = st.radio("Multiple Lines", ("Yes", "No", "No phone service"), help="Does the customer have multiple lines?")
    internet_service = st.radio("Internet Service", ("DSL", "Fiber optic", "No"), help="Type of internet service")

with col3:
    st.subheader("Online Services")
    online_security = st.radio("Online Security", ("No", "Yes", "No internet service"), help="Does the customer have online security?")
    online_backup = st.radio("Online Backup", ("No", "Yes", "No internet service"), help="Does the customer have online backup?")
    device_protection = st.radio("Device Protection", ("No", "Yes", "No internet service"), help="Does the customer have device protection?")

with col4:
    st.subheader("Streaming & Support")
    tech_support = st.radio("Tech Support", ("No", "Yes", "No internet service"), help="Does the customer have tech support?")
    streaming_tv = st.radio("Streaming TV", ("No", "Yes", "No internet service"), help="Does the customer stream TV?")
    streaming_movies = st.radio("Streaming Movies", ("No", "Yes", "No internet service"), help="Does the customer stream movies?")

with col5:
    st.subheader("Billing & Contract")
    contract = st.radio("Contract", ("Month-to-month", "One year", "Two year"), help="Type of contract")
    paperless_billing = st.radio("Paperless Billing", ("Yes", "No"), help="Does the customer use paperless billing?")
    payment_method = st.radio("Payment Method", (
        "Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"
    ), help="Payment method used by the customer")

# Additional numerical inputs
st.markdown("---")
st.subheader("Charges & Tenure")
tenure = st.slider("Tenure (Months)", min_value=1, max_value=72, value=12, help="Number of months the customer has been with the company")
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=79.85, help="Monthly charges for the customer")
total_charges = st.number_input("Total Charges", min_value=0.0, value=942.25, help="Total charges incurred by the customer")

# Predict button
if st.button("🔮 Predict Churn", key="predict-button"):
    with st.spinner("Predicting... Please wait"):
        data = {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": int(tenure),
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": float(monthly_charges),
            "TotalCharges": float(total_charges)
        }

        try:
            response = requests.post(API_URL, json=data)
            response.raise_for_status()
            prediction = response.json()
            churn_probability = prediction.get("churn_probability", 0)
            churn_result = "Churn" if prediction.get("churn_prediction", False) else "No Churn"
            
            st.success(f"Prediction: {churn_result}")
            st.success(f"Churn Probability: {churn_probability:.2f}")
            
            # Visualization
            # labels = ["No Churn", "Churn"]
            # values = [1 - churn_probability, churn_probability]
            # fig, ax = plt.subplots(figsize=(4, 4))
            # ax.pie(values, labels=labels, autopct='%1.1f%%', colors=['green', 'red'], startangle=90)
            # ax.set_title("Churn Probability Distribution")
            # st.pyplot(fig)
            st.markdown("---")
            fig = px.pie(
                names=["No Churn", "Churn"],
                values=[1 - churn_probability, churn_probability],
                title="Churn Probability Distribution",
                color=["No Churn", "Churn"],
                color_discrete_map={"No Churn": "green", "Churn": "red"}
            )
            st.plotly_chart(fig, use_container_width=True)

        except requests.exceptions.RequestException as e:
            st.error(f"API Request failed: {e}")

# JavaScript to trigger the button click
st.markdown(
    """
    <script>
    function triggerButton() {
        document.getElementById('predict-button').click();
    }
    </script>
    """,
    unsafe_allow_html=True,
)