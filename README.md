# Predictive-Analysis-for-Manufacturing-Operations
# Predictive Analysis for Manufacturing Operations

## Objective
This project implements a RESTful API to predict machine downtime or production defects using a supervised machine learning model. The API supports uploading data, training a model, and making predictions for informed decision-making.

---

## Features
- **Upload Manufacturing Data**: Upload datasets containing machine-specific information.
- **Train Predictive Models**: Use the provided dataset to train a model that predicts machine downtime or defects.
- **Make Predictions**: Receive predictions based on provided input parameters, such as temperature and runtime.

---

## Prerequisites
Ensure the following tools and packages are installed:
- **Python 3.8 or later**
- **pip** (Python package manager)
- **Postman** or **cURL** (for testing the API)

---

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/vignesh3003/Predictive-Analysis-for-Manufacturing-Operations.git
   cd Predictive-Analysis-for-Manufacturing-Operations
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the API Server**:
   ```bash
   python app.py
   ```

5. **Access the API**:
   The server runs locally at `http://127.0.0.1:5000`.

---

## API Endpoints

### 1. **Upload Data**
**Endpoint**: `POST /upload`  
**Description**: Upload a CSV file containing manufacturing data.  

**Example Request** (using cURL):
```bash
curl -X POST -F "file=@sample.csv" http://127.0.0.1:5000/upload
```

**Example Response**:
```json
{
  "message": "File uploaded successfully."
}
```

---

### 2. **Train Model**
**Endpoint**: `POST /train`  
**Description**: Train a machine learning model on the uploaded dataset.  

**Example Request**:
```bash
curl -X POST http://127.0.0.1:5000/train
```

**Example Response**:
```json
{
  "message": "Model trained successfully.",
  "accuracy": 0.89,
  "f1_score": 0.85
}
```

---

### 3. **Make Predictions**
**Endpoint**: `POST /predict`  
**Description**: Predict machine downtime or defects based on input parameters.  

**Example Request**:
```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"Temperature": 80, "Run_Time": 120}' \
     http://127.0.0.1:5000/predict
```

**Example Response**:
```json
{
  "Downtime": "Yes",
  "Confidence": 0.85
}
```

---

## Example Dataset
The dataset should include the following columns:
- `Machine_ID`
- `Temperature`
- `Run_Time`
- `Downtime_Flag`

If no dataset is available, synthetic data with these columns can be generated.

---

## Testing the API
- Use **Postman** or **cURL** to test all endpoints.
- Ensure the API server is running locally before sending requests.

---

## Future Improvements
- Add advanced machine learning models for better predictions.
- Integrate a database for persistent data storage.
- Deploy the API to cloud platforms such as AWS, Azure, or Heroku.

---

## Repository Contents
- **`app.py`**: Main API server code.
- **`requirements.txt`**: List of dependencies.
- **Sample Dataset**: Example data for testing the API.
- **README.md**: Project documentation (this file).

---

For any questions or issues, feel free to contact [Vignesh](mailto:vignesh@example.com).

