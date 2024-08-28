import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

# Load the trained model from the pickle file
model_path = 'rent_price_model.pkl'
model = joblib.load(model_path)

# Define the request body structure
class HouseData(BaseModel):
    Location: str
    Area: float
    Bedrooms: int
    Parking: int

# Initialize FastAPI
app = FastAPI()

# Define the prediction endpoint
@app.post("/predict")
def predict_rent_price(house_data: HouseData):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([house_data.dict()])
    
    # Make prediction
    try:
        prediction = model.predict(input_df)
        return {"predicted_rent_price": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Optional: You can add an evaluation endpoint if needed, or any other endpoints

# Run the FastAPI application
# Uncomment the following lines if you want to run the application directly
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
