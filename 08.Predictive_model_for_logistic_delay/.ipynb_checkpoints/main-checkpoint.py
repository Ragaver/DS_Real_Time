from flask import Flask, render_template,request
import joblib
import pandas as pd

preprocessor = joblib.load('preprocessor.pkl')
model = joblib.load('model.pkl')

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict",methods = ['POST'])
def predict():
    try:
        Asset_ID = request.form['Asset_ID']
        Inventory_Level = request.form['Inventory_Level']
        Shipment_Status = request.form['Shipment_Status']
        Temperature = float(request.form['Temperature'])
        Humidity = float(request.form['Humidity'])
        Traffic_Status = request.form['Traffic_Status']
        Waiting_Time = request.form['Waiting_Time']
        User_Transaction_Amount = request.form['User_Transaction_Amount']
        User_Purchase_Frequency = request.form['User_Purchase_Frequency']
        Logistics_Delay_Reason = request.form['Logistics_Delay_Reason']
        Asset_Utilization = float(request.form['Asset_Utilization'])
        Demand_Forecast = request.form['Demand_Forecast']

        input_data = pd.DataFrame([[Asset_ID,Inventory_Level,Shipment_Status,Temperature,Humidity,Traffic_Status,Waiting_Time,
                                    User_Transaction_Amount,User_Purchase_Frequency,Logistics_Delay_Reason,Asset_Utilization,Demand_Forecast
                                    ]],columns = ['Asset_ID','Inventory_Level','Shipment_Status','Temperature','Humidity','Traffic_Status','Waiting_Time',
                                    'User_Transaction_Amount','User_Purchase_Frequency','Logistics_Delay_Reason','Asset_Utilization','Demand_Forecast'
                                    ])
        pre = preprocessor.transform(input_data)
        pred = model.predict(pre)[0]

        return render_template('index.html',prediction = f'Prediction: {pred}')
    except Exception as e:
        return render_template('index.html',prediction = f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug = True)
