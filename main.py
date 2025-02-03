from fastapi import FastAPI
from fastapi import HTTPException
import pandas as pd
import joblib
from pydantic import BaseModel, constr, Field
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model and scaler
model = joblib.load("models/churn_model.pkl")
encoding_files = joblib.load("models/encoding_files.pkl")
scaler = joblib.load("models/scaler.pkl")

def preprocess_input(df,encoding_files):
    df = df.map(lambda x: x.lower() if isinstance(x, str) else x)
    encoders = encoding_files['label_encoders']
    columns = encoding_files['columns']
    df['gender'] = encoders['gender'].transform(df['gender'])
    df['Contract'] = encoders['contract'].transform(df['Contract'])
    df['InternetService'] = encoders['internet'].transform(df['InternetService'])
    df['PaymentMethod'] = encoders['payment'].transform(df['PaymentMethod'])
    for col in columns['n_y']:
        df[col] = encoders['n_y'].transform(df[col])
    for col in columns['n_y_nps']:
        df[col] = encoders['n_y_nps'].transform(df[col])
    for col in columns['n_y_nis']:
        df[col] = encoders['n_y_nis'].transform(df[col])
    
    numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    df[numerical_cols] = scaler.transform(df[numerical_cols])

    columns = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService", 
                "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", 
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", 
                "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"]
        
    df = df[columns]
    return df


# Define the input data model
class CustomerData(BaseModel):
    gender: str = Field(..., example="Male", description="Gender of the customer (Male/Female)")             # Original form: "Male" or "Female"
    SeniorCitizen: str      # Original form: "Yes" or "No"
    Partner: str            # Original form: "Yes" or "No"
    Dependents: str         # Original form: "Yes" or "No"
    tenure: int = Field(..., example=12, ge=0, description="Number of months the customer has been with the company")             # No of months
    PhoneService: str       # Original form: "Yes" or "No"
    MultipleLines: str      # Original form: "Yes", "No", or "No phone service"
    InternetService: str = Field(..., example="Fiber optic", description="Type of internet service (DSL/Fiber optic/No)")    # Original form: "DSL", "Fiber optic", or "No"
    OnlineSecurity: str     # Original form: "Yes", "No", or "No internet service"
    OnlineBackup: str       # Original form: "Yes", "No", or "No internet service"
    DeviceProtection: str   # Original form: "Yes", "No", or "No internet service"
    TechSupport: str        # Original form: "Yes", "No", or "No internet service"
    StreamingTV: str        # Original form: "Yes", "No", or "No internet service"
    StreamingMovies: str    # Original form: "Yes", "No", or "No internet service"
    Contract: str = Field(..., example="Month-to-month", description="Customer's contract type (Month-to-month/One year/Two year)")           # Original form: "Month-to-month", "One year", or "Two year"
    PaperlessBilling: str   # Original form: "Yes" or "No"
    PaymentMethod: str      # Original form: "Electronic check", "Mailed check", "Bank transfer (automatic)", or "Credit card (automatic)"
    MonthlyCharges: float = Field(..., example=79.85, ge=0, description="Monthly charges in USD")
    TotalCharges: float = Field(..., example=942.25, ge=0, description="Total amount charged to the customer")
    
    class Config:
        schema_extra = {
            "example": {
                "gender": "Male",
                "tenure": 12,
                "MonthlyCharges": 79.85,
                "TotalCharges": 942.25,
                "Contract": "Month-to-month",
                "InternetService": "Fiber optic"
            }
        }

# Initialize FastAPI
app = FastAPI(title="Customer Churn Prediction API", description="Predicts whether a customer will churn based on provided features.", version="1.0.0")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Define the prediction endpoint
@app.post("/predict", summary="Predict Churn", description="Predicts whether a customer is likely to churn based on input data.")
async def predict(data: CustomerData):
    logger.info(f"Received prediction request: {data.dict()}")
    try:
        # Convert input data to DataFrame
        input_data = pd.DataFrame([data.dict()])
        processed_data = preprocess_input(input_data,encoding_files)

        # Make prediction
        prediction = model.predict(processed_data)
        churn_prob = model.predict_proba(processed_data)[0][1]

        logger.info(f"Model made a prediction: prediction ={prediction}, churn_prob={churn_prob}")

        # Return the result
        return {"churn_prediction": int(prediction[0]), "churn_probability": float(churn_prob), "result":"Churn" if int(prediction[0]) == 1 else "Not Churn"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing request: {str(e)}")