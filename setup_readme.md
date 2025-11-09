# Health Insurance Prediction System - Complete Setup Guide

## 📋 Overview
Professional health insurance prediction system with ML-powered eligibility and cost prediction for hospital use.

## 🗂️ Files Provided

1. **Fixed_Health_Insurance_Prediction_System.ipynb** - Jupyter notebook with complete ML pipeline
2. **insurance_app.html** - Standalone web application (works immediately)
3. **app.py** - Flask backend API (optional, for production)
4. **README.md** - This file

## 🚀 Quick Start (Web App Only)

### Option 1: Direct Browser Use (Simplest)
1. Open `insurance_app.html` in any web browser
2. Fill in patient information
3. Click "Predict" to get instant results
4. **No installation required!**

## 🔬 Full ML Setup (With Real Models)

### Step 1: Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib flask flask-cors
```

### Step 2: Run the Jupyter Notebook
```bash
jupyter notebook Fixed_Health_Insurance_Prediction_System.ipynb
```

Run all cells to:
- Generate training data
- Train ML models
- Save models to disk (`hybrid_insurance_predictor.pkl`)

### Step 3: Start Flask Backend (Optional)
```bash
python app.py
```

Server will start at `http://localhost:5000`

### Step 4: Update Web App to Use Backend
Modify the `insurance_app.html` to call the API:

```javascript
// Replace the predictInsurance() function with this:
async function predictInsurance() {
    const data = {
        age: parseInt(document.getElementById('age').value),
        gender: document.getElementById('gender').value,
        bmi: parseFloat(document.getElementById('bmi').value),
        children: parseInt(document.getElementById('children').value),
        smoker: document.getElementById('smoker').value,
        region: document.getElementById('region').value,
        pre_existing_conditions: document.getElementById('pre_existing_conditions').value,
        employment_status: document.getElementById('employment_status').value,
        annual_income: parseInt(document.getElementById('annual_income').value),
        previous_claims: parseInt(document.getElementById('previous_claims').value)
    };

    try {
        const response = await fetch('http://localhost:5000/api/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        // Update UI with results
        // ... (rest of display code)
    } catch (error) {
        console.error('Prediction error:', error);
        alert('Error connecting to prediction service');
    }
}
```

## 📊 Model Details

### Eligibility Prediction
- **Algorithm**: Stacking Classifier
- **Base Models**: Random Forest, Logistic Regression
- **Accuracy**: ~90%
- **Output**: Binary (Eligible/Not Eligible) + Confidence Score

### Cost Prediction
- **Algorithm**: Stacking Regressor
- **Base Models**: Random Forest, Gradient Boosting
- **R² Score**: ~0.90
- **Output**: Annual premium in USD

### Hybrid Approach
- Eligibility probability used as feature for cost prediction
- Two-stage pipeline with cross-validation
- Feature augmentation to link classification and regression

## 🏥 Hospital Deployment

### Security Considerations
1. **HTTPS**: Use SSL certificates for production
2. **Authentication**: Add user login system
3. **Data Privacy**: Implement HIPAA-compliant data handling
4. **Audit Logs**: Track all predictions for compliance

### Production Checklist
- [ ] Deploy Flask app on production server (AWS, Azure, GCP)
- [ ] Set up database for logging predictions
- [ ] Configure firewall and security groups
- [ ] Implement rate limiting
- [ ] Add monitoring and alerting
- [ ] Regular model retraining schedule
- [ ] Backup and disaster recovery plan

## 🔧 API Documentation

### Endpoint: `/api/predict`
**Method**: POST

**Request Body**:
```json
{
  "age": 35,
  "gender": "Male",
  "bmi": 24.5,
  "children": 2,
  "smoker": "No",
  "region": "Northeast",
  "pre_existing_conditions": "None",
  "employment_status": "Employed",
  "annual_income": 75000,
  "previous_claims": 1
}
```

**Response**:
```json
{
  "eligible": 1,
  "eligibility_probability": 0.92,
  "predicted_cost": 18450.50,
  "monthly_cost": 1537.54,
  "model_version": "ml-hybrid-v1.0"
}
```

### Endpoint: `/api/batch_predict`
**Method**: POST

Process multiple patients at once.

### Endpoint: `/api/health`
**Method**: GET

Check API status.

## 📈 Model Retraining

Update models with new data:

```python
# Load new data
new_data = pd.read_csv('new_patient_data.csv')

# Retrain
predictor.eligibility_model.fit(X_new, y_eligibility_new)
predictor.cost_model.fit(X_augmented_new, y_cost_new)

# Save
joblib.dump(predictor, 'hybrid_insurance_predictor_v2.pkl')
```

## 🐛 Troubleshooting

**Issue**: Model file not found
- **Solution**: Run the Jupyter notebook first to generate models

**Issue**: CORS errors in browser
- **Solution**: Install flask-cors: `pip install flask-cors`

**Issue**: Prediction returns error
- **Solution**: Check all required fields are provided and in correct format

## 📞 Support

For issues or questions:
- Check model logs in notebook output
- Verify all dependencies are installed
- Ensure data formats match expected types

## 🔄 Version History

- **v1.0.0** (2025-01-07): Initial release
  - Hybrid stacking ensemble
  - Web interface
  - REST API
  - Rule-based fallback

## 📝 License

For hospital and healthcare use. Ensure compliance with local regulations (HIPAA, GDPR, etc.)

---

**Note**: This is a demonstration system. For production use in hospitals, please consult with healthcare IT compliance teams and conduct thorough validation studies.
