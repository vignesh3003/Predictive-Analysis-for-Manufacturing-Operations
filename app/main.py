from fastapi import FastAPI, UploadFile, File, HTTPException
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import pandas as pd
import joblib
import io
import os

app = FastAPI()

# Global variables
model = None  # The predictive model
scaler = None  # Feature scaler
MODEL_PATH = "model.joblib"  # Path to save the trained model
SCALER_PATH = "scaler.joblib"  # Path to save the scaler

@app.get("/")
def read_root():
    """
    Root endpoint with API description.
    """
    return {
        "message": "Welcome to the FastAPI Machine Learning API.",
        "endpoints": {
            "Train Endpoint": "/train",
            "Predict Endpoint": "/predict"
        }
    }

@app.post("/train")
async def train_model(file: UploadFile = File(...)):
    """
    Train the model using the uploaded CSV file.
    """
    global model, scaler
    allowed_content_types = ["text/csv", "application/csv", "application/vnd.ms-excel"]
    if not any(file.content_type.startswith(ct) for ct in allowed_content_types):
        raise HTTPException(status_code=400, detail=f"Invalid content type: {file.content_type}")

    try:
        contents = await file.read()
        data = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        # Validate required columns
        required_columns = {"Temperature", "Run_Time", "Downtime_Flag"}
        if not required_columns.issubset(data.columns):
            raise HTTPException(status_code=400, detail=f"Dataset must include columns: {required_columns}")

        # Drop unnecessary columns
        data = data.drop(columns=[col for col in data.columns if 'Machine' in col], errors='ignore')

        # Convert categorical data to numerical
        categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
        if categorical_columns:
            label_encoder = LabelEncoder()
            for col in categorical_columns:
                data[col] = label_encoder.fit_transform(data[col].astype(str))

        # Handle missing values
        data.fillna(data.mean(), inplace=True)

        # Oversample to balance classes
        X = data[["Temperature", "Run_Time"]]
        y = data["Downtime_Flag"]
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_resampled)

        # Save the scaler
        joblib.dump(scaler, SCALER_PATH)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42)

        # Hyperparameter tuning
        param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None]}
        grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
        grid_search.fit(X_train, y_train)

        model = grid_search.best_estimator_

        # Save the trained model
        joblib.dump(model, MODEL_PATH)

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        return {
            "message": "Model trained successfully.",
            "accuracy": round(accuracy, 4)  # Round accuracy to 4 decimal places
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")

@app.post("/predict")
def predict(input_data: dict):
    """
    Make predictions based on input JSON data.
    """
    global model, scaler
    try:
        # Ensure model and scaler are loaded
        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            raise HTTPException(status_code=400, detail="Model and scaler are not trained yet. Please train them first.")
        if model is None:
            model = joblib.load(MODEL_PATH)
        if scaler is None:
            scaler = joblib.load(SCALER_PATH)

        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Validate input features
        required_features = ["Temperature", "Run_Time"]
        if not set(required_features).issubset(input_df.columns):
            raise HTTPException(status_code=400, detail=f"Missing features in input data: {required_features}")

        # Scale input data
        input_scaled = scaler.transform(input_df[required_features])

        # Make predictions
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)
        confidence = max(prediction_proba[0])

        # Log raw probabilities for debugging
        print(f"Input: {input_data}")
        print(f"Probabilities: {prediction_proba}")

        return {
            "Downtime": "Yes" if prediction[0] == 1 else "No",
            "Confidence": round(confidence, 4)  # Round confidence to 4 decimal places
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
