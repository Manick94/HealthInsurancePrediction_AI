"""
Flask Backend API for Health Insurance Prediction System
Run this file to create a REST API that the web app can connect to
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for web app integration

# Load models (create these using the notebook first)
try:
    predictor = joblib.load('hybrid_insurance_predictor.pkl')
    MODEL_LOADED = True
    print("✓ ML Model loaded successfully")
except:
    MODEL_LOADED = False
    print("⚠ Warning: ML model not found. Using rule-based predictions.")

# Rule-based prediction fallback
def rule_based_prediction(data):
    """Fallback prediction when ML model is not available"""
    
    # Eligibility calculation
    elig_score = 0
    if data['age'] < 60: elig_score += 2
    if data['smoker'] == 'No': elig_score += 3
    if data['pre_existing_conditions'] == 'None': elig_score += 3
    if data['bmi'] < 30: elig_score += 1
    if data['annual_income'] > 30000: elig_score += 1
    
    is_eligible = elig_score >= 6
    confidence = min(95, 60 + elig_score * 5)
    
    # Cost calculation
    cost = 5000
    cost += (data['age'] - 18) * 250
    cost += data['bmi'] * 100
    cost += data['children'] * 1500
    if data['smoker'] == 'Yes': cost *= 1.6
    if data['pre_existing_conditions'] != 'None': cost *= 1.3
    cost += data['previous_claims'] * 800
    cost = max(1000, round(cost, 2))
    
    # Risk factors
    risks = []
    if data['smoker'] == 'Yes':
        risks.append({'factor': 'Smoking Status', 'severity': 'High', 'impact': '+60% cost'})
    if data['pre_existing_conditions'] != 'None':
        risks.append({'factor': f"Pre-existing: {data['pre_existing_conditions']}", 
                     'severity': 'High', 'impact': '+30% cost'})
    if data['bmi'] > 30:
        risks.append({'factor': f"High BMI ({data['bmi']})", 
                     'severity': 'Medium', 'impact': 'Increased risk'})
    if data['age'] > 50:
        risks.append({'factor': f"Age ({data['age']} years)", 
                     'severity': 'Medium', 'impact': '+$250/year'})
    if data['previous_claims'] > 3:
        risks.append({'factor': f"{data['previous_claims']} previous claims", 
                     'severity': 'Medium', 'impact': f"+${data['previous_claims']*800}"})
    
    return {
        'eligible': int(is_eligible),
        'eligibility_probability': confidence / 100,
        'predicted_cost': cost,
        'monthly_cost': round(cost / 12, 2),
        'risk_factors': risks,
        'model_version': 'rule-based-v1.0'
    }

@app.route('/')
def home():
    """Serve the web application"""
    with open('insurance_app.html', 'r') as f:
        return f.read()

@app.route('/api/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['age', 'gender', 'bmi', 'children', 'smoker', 'region',
                          'pre_existing_conditions', 'employment_status', 
                          'annual_income', 'previous_claims']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Convert to proper types
        patient_data = {
            'age': int(data['age']),
            'gender': str(data['gender']),
            'bmi': float(data['bmi']),
            'children': int(data['children']),
            'smoker': str(data['smoker']),
            'region': str(data['region']),
            'pre_existing_conditions': str(data['pre_existing_conditions']),
            'employment_status': str(data['employment_status']),
            'annual_income': int(data['annual_income']),
            'previous_claims': int(data['previous_claims'])
        }
        
        # Make prediction
        if MODEL_LOADED:
            # Use ML model
            df = pd.DataFrame([patient_data])
            results = predictor.predict(df)
            
            prediction = {
                'eligible': int(results['eligible'].iloc[0]),
                'eligibility_probability': float(results['eligibility_probability'].iloc[0]),
                'predicted_cost': float(results['predicted_cost'].iloc[0]),
                'monthly_cost': float(results['predicted_cost'].iloc[0] / 12),
                'model_version': 'ml-hybrid-v1.0'
            }
        else:
            # Use rule-based fallback
            prediction = rule_based_prediction(patient_data)
        
        return jsonify(prediction)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction for multiple patients"""
    try:
        data = request.json
        patients = data.get('patients', [])
        
        if not patients:
            return jsonify({'error': 'No patient data provided'}), 400
        
        results = []
        for patient in patients:
            if MODEL_LOADED:
                df = pd.DataFrame([patient])
                pred = predictor.predict(df)
                results.append({
                    'patient_id': patient.get('id', None),
                    'eligible': int(pred['eligible'].iloc[0]),
                    'predicted_cost': float(pred['predicted_cost'].iloc[0])
                })
            else:
                pred = rule_based_prediction(patient)
                results.append({
                    'patient_id': patient.get('id', None),
                    'eligible': pred['eligible'],
                    'predicted_cost': pred['predicted_cost']
                })
        
        return jsonify({'predictions': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/statistics', methods=['GET'])
def statistics():
    """Get model statistics and performance metrics"""
    try:
        if MODEL_LOADED:
            stats = {
                'model_type': 'Hybrid Stacking Ensemble',
                'eligibility_model': 'Stacking Classifier (RF + LR)',
                'cost_model': 'Stacking Regressor (RF + GB)',
                'features': 10,
                'status': 'operational'
            }
        else:
            stats = {
                'model_type': 'Rule-based',
                'status': 'fallback mode'
            }
        
        return jsonify(stats)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL_LOADED,
        'version': '1.0.0'
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🏥 Health Insurance Prediction API")
    print("="*60)
    print(f"Model Status: {'✓ Loaded' if MODEL_LOADED else '⚠ Using fallback'}")
    print("\nAvailable Endpoints:")
    print("  POST /api/predict        - Single prediction")
    print("  POST /api/batch_predict  - Batch predictions")
    print("  GET  /api/statistics     - Model statistics")
    print("  GET  /api/health         - Health check")
    print("\nStarting server on http://localhost:5300")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5300)
