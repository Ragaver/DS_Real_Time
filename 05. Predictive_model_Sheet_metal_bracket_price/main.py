# Import required packages
from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd  # ✅ Import Pandas to handle DataFrames

# Load the trained pipeline model
with open('model.pkl', 'rb') as f:
    pipeline = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
@app.route("/Predict", methods=['POST'])
def Predict():
    try:
        # Get form inputs
        input_data = {
            "Plant_code": [request.form['Plant_code']],
            "Supplier": [request.form['Supplier']],
            "Material_user_country": [request.form['Material_user_country']],
            "Item_weight_kg": [float(request.form['Item_weight_kg'])],
            "One_Year_forecast_qty": [int(request.form['One_Year_forecast_qty'])],
            "12_month_forecast_spend": [float(request.form['12_month_forecast_spend'])]
        }

        # ✅ Convert to DataFrame (Fix for the erro
        input_df = pd.DataFrame(input_data)

        # ✅ Preprocess input
        preprocessed_data = pipeline.named_steps['preprocessor'].transform(input_df)

        # ✅ Predict using the regression model
        Prediction_value = pipeline.named_steps['model'].predict(preprocessed_data)[0]

        # ✅ Return result to HTML page
        return render_template('index.html', Prediction=f"Predicted Price (JPY): {Prediction_value:.2f}")

    except Exception as e:
        return render_template('index.html', Prediction=f"Error: {str(e)}")

# Run the Flask application
if __name__ == "__main__":
    app.run(debug=True)
