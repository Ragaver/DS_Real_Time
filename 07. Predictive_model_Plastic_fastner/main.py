from flask import Flask,render_template,request
import joblib
import pandas as pd

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict",methods = ['POST'])
def predict():
    try:
        preprocessor = joblib.load('preprocessor.pkl')
        model = joblib.load('model.pkl')

        Supplier_region = request.form.get('Supplier_region')
        Material_user = request.form.get('Material_user')
        Item_weight_gm = float(request.form.get('Item_weight_gm'))

        input_data = pd.DataFrame([[Supplier_region,Material_user,Item_weight_gm]],
                                columns = ['Supplier_region','Material_user','Item_weight_gm'])

        pre = preprocessor.transform(input_data)

        pred = model.predict(pre)[0]
        return render_template('index.html',prediction = f'Prediction is {pred:.2f}')
    
    except Exception as e:
        return render_template('index.html',prediction = f'Error: {e}')

if __name__ == '__main__':
    app.run(debug = True)

