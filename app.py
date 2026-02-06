"""
Viva Evaluation System - Flask Application
Serves both backend ML inference and HTML frontend
"""

import os
import pickle
from flask import Flask, render_template, request, jsonify
from flask_swagger_ui import get_swaggerui_blueprint
import numpy as np
import pandas as pd

app = Flask(__name__)

# Swagger UI configuration
SWAGGER_URL = '/apidocs'
API_URL = '/swagger.json'

# Create Swagger UI blueprint
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Viva Evaluation System API"
    }
)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# Global variable to store the loaded model
model = None


class CompatUnpickler(pickle.Unpickler):
    """Custom unpickler to handle numpy version compatibility"""
    def find_class(self, module, name):
        # Fix numpy._core compatibility issues
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        elif module == 'numpy.core.multiarray' and name == '_reconstruct':
            module = 'numpy'
        return super().find_class(module, name)


def load_model():
    """Load the trained ML model from pickle file"""
    global model
    
    model_path = 'final_model.pkl'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    
    try:
        # Try joblib first (preferred for scikit-learn)
        try:
            import joblib
            loaded_model = joblib.load(model_path)
            if hasattr(loaded_model, 'predict'):
                model = loaded_model
                print(f"✓ Model loaded from {model_path} (joblib)")
                return
        except:
            pass
        
        # Try pickle with compatibility fix
        with open(model_path, 'rb') as f:
            unpickler = CompatUnpickler(f)
            loaded_obj = unpickler.load()
        
        # Check if it's a model
        if hasattr(loaded_obj, 'predict'):
            model = loaded_obj
            print(f"✓ Model loaded from {model_path} (pickle)")
            return
        
        # If it's a DataFrame, create a model from it
        if hasattr(loaded_obj, 'columns') and hasattr(loaded_obj, 'values'):
            import pandas as pd
            from sklearn.linear_model import LinearRegression
            
            df = loaded_obj
            # Check if DataFrame has the right columns
            required_cols = ['q1_score', 'q2_score', 'q3_score', 'q4_score', 'q5_score']
            if 'predicted_total_score' in df.columns and all(col in df.columns for col in required_cols):
                print(f"⚠ File contains DataFrame with predictions. Training model from data...")
                X = df[required_cols].values
                y = df['predicted_total_score'].values
                
                # Train a simple linear model
                model = LinearRegression()
                model.fit(X, y)
                print(f"✓ Model trained from {model_path} DataFrame")
                return
        
        raise Exception(f"File contains {type(loaded_obj).__name__}, not a model")
            
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")


def calculate_grade(total_score):
    """
    Calculate grade based on total score
    
    Grade Logic:
    - total_score >= 40 → A
    - total_score >= 30 → B
    - total_score >= 25 → C
    - else → Fail
    """
    if total_score >= 40:
        return "A"
    elif total_score >= 30:
        return "B"
    elif total_score >= 25:
        return "C"
    else:
        return "Fail"


@app.route('/swagger.json')
def swagger():
    """Swagger API specification"""
    return jsonify({
        "swagger": "2.0",
        "info": {
            "title": "Viva Evaluation System API",
            "description": "API for predicting total viva scores and grades from individual question scores using machine learning",
            "version": "1.0.0"
        },
        "basePath": "/",
        "schemes": ["http", "https"],
        "consumes": ["application/json"],
        "produces": ["application/json"],
        "paths": {
            "/predict": {
                "post": {
                    "tags": ["Predictions"],
                    "summary": "Predict total score and grade",
                    "description": "Uses machine learning model to predict total score from 5 individual question scores and assigns a grade",
                    "consumes": ["application/json"],
                    "produces": ["application/json"],
                    "parameters": [
                        {
                            "in": "body",
                            "name": "body",
                            "description": "Question scores for prediction",
                            "required": True,
                            "schema": {
                                "type": "object",
                                "required": ["q1_score", "q2_score", "q3_score", "q4_score", "q5_score"],
                                "properties": {
                                    "q1_score": {
                                        "type": "number",
                                        "format": "float",
                                        "description": "Score for question 1",
                                        "example": 8.0
                                    },
                                    "q2_score": {
                                        "type": "number",
                                        "format": "float",
                                        "description": "Score for question 2",
                                        "example": 7.0
                                    },
                                    "q3_score": {
                                        "type": "number",
                                        "format": "float",
                                        "description": "Score for question 3",
                                        "example": 9.0
                                    },
                                    "q4_score": {
                                        "type": "number",
                                        "format": "float",
                                        "description": "Score for question 4",
                                        "example": 8.0
                                    },
                                    "q5_score": {
                                        "type": "number",
                                        "format": "float",
                                        "description": "Score for question 5",
                                        "example": 7.0
                                    }
                                }
                            }
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Successful prediction",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "total_score": {
                                        "type": "number",
                                        "format": "float",
                                        "description": "Predicted total score",
                                        "example": 39.0
                                    },
                                    "grade": {
                                        "type": "string",
                                        "description": "Assigned grade (A, B, C, or Fail)",
                                        "example": "B"
                                    }
                                }
                            }
                        },
                        "400": {
                            "description": "Bad request - invalid or missing input",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "error": {
                                        "type": "string",
                                        "example": "Missing required fields: q1_score, q2_score"
                                    }
                                }
                            }
                        },
                        "500": {
                            "description": "Server error - model not loaded or prediction failed",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "error": {
                                        "type": "string",
                                        "example": "Model not loaded. Please restart the application."
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    })


@app.route('/')
def index():
    """Serve the HTML frontend"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict total score and grade from question scores
    
    Expected JSON input:
    {
        "q1_score": float,
        "q2_score": float,
        "q3_score": float,
        "q4_score": float,
        "q5_score": float
    }
    
    Returns JSON:
    {
        "total_score": float,
        "grade": string
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Validate required fields
        required_fields = ['q1_score', 'q2_score', 'q3_score', 'q4_score', 'q5_score']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                "error": f"Missing required fields: {', '.join(missing_fields)}"
            }), 400
        
        # Extract and validate scores
        scores = []
        for field in required_fields:
            try:
                score = float(data[field])
                if score < 0:
                    return jsonify({
                        "error": f"Invalid value for {field}: must be non-negative"
                    }), 400
                scores.append(score)
            except (ValueError, TypeError):
                return jsonify({
                    "error": f"Invalid value for {field}: must be a number"
                }), 400
        
        # Ensure model is loaded
        if model is None:
            return jsonify({
                "error": "Model not loaded. Please restart the application."
            }), 500
        
        # final_model.pkl is a Pipeline (ColumnTransformer + RandomForest) that expects
        # 6 features: q1_score, q2_score, q3_score, q4_score, q5_score, total_score,
        # and predicts grade. Compute total_score as sum of the 5 question scores.
        total_score = sum(scores)
        model_input = pd.DataFrame(
            [scores + [total_score]],
            columns=['q1_score', 'q2_score', 'q3_score', 'q4_score', 'q5_score', 'total_score']
        )
        
        try:
            grade = model.predict(model_input)[0]
        except Exception as e:
            return jsonify({
                "error": f"Prediction error: {str(e)}"
            }), 500
        
        # Return prediction result (total_score from sum; grade from pipeline)
        return jsonify({
            "total_score": round(total_score, 2),
            "grade": str(grade)
        })
    
    except Exception as e:
        return jsonify({
            "error": f"Server error: {str(e)}"
        }), 500


if __name__ == '__main__':
    # Load model on startup
    try:
        load_model()
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        print("Application will start but predictions will fail until model is available.")
        model = None
    
    # Run Flask app
    print("Starting Viva Evaluation System...")
    print("Access the application at: http://127.0.0.1:5000")
    print("Swagger API documentation at: http://127.0.0.1:5000/apidocs")
    app.run(debug=True, host='127.0.0.1', port=5000)
