from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load('model.pkl','rb')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/' ,methods="POST" )
def predict():
    age     = np.reshape(float(request.values['feature_age'])       ,(-1,1))
    loan    = np.reshape(float(request.values['feature_loan'])      ,(-1,1))
    house   = np.reshape(float(request.values['feature_housing'])   ,(-1,1))
    default = np.reshape(float(request.values['feature_default'])   ,(-1,1))
    job     = np.reshape(float(request.values['feature_job'])       ,(-1,1))
    edu     = np.reshape(float(request.values['feature_education']) ,(-1,1))
    marital = np.reshape(float(request.values['feature_marital'])   ,(-1,1))
    month   = np.reshape(float(request.values['feature_month'])     ,(-1,1))

    output = model.predict(age, loan, house, default, job, edu, marital, month)
    output = output.item
    output = round(output,2)

    msg = 'success' if output == 1 else 'Failed' 

    return render_template('index.html', prediction_text='The prediction is '.format(msg))

if __name__ == '__main__':
    app.run(debug=True)

