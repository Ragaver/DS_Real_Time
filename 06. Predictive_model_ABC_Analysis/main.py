from flask import Flask,render_template,request
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
        Product_MRP = float(request.form['Product_MRP'])
        Product_Cost = float(request.form['Product_Cost'])
        Profit_Margin = float(request.form['Profit_Margin'])
        
        input_data = pd.DataFrame([[Product_MRP,Product_Cost,Profit_Margin]],
        columns = ['Product_MRP','Product_Cost','Profit_Margin'])

        pre = preprocessor.transform(input_data)

        pred = model.predict(pre)[0]
        return render_template('index.html',prediction = f'Prediction {pred}')
    except Exception as e:
        return render_template('index.html',prediction = f'Prediction {str(e)}')

if __name__== '__main__':
    app.run(debug = True)

