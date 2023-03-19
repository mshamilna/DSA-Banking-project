from flask import Flask,render_template,request
import pickle
import numpy as np
import pandas as pd

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods =['POST'])
def predict(): 
    age     = np.reshape(float(request.values['feature_age'])       ,(-1,1))
    job     = np.reshape(float(request.values['feature_job'])       ,(-1,1))
    marital = np.reshape(float(request.values['feature_marital'])   ,(-1,1))
    edu     = np.reshape(float(request.values['feature_education']) ,(-1,1))
    default = np.reshape(float(request.values['feature_default'])   ,(-1,1)) 
    balance = np.reshape(float(request.values['feature_balance'])   ,(-1,1))
    house   = np.reshape(float(request.values['feature_housing'])   ,(-1,1))
    loan    = np.reshape(float(request.values['feature_loan'])      ,(-1,1))
    contact = np.reshape(float(request.values['feature_contact'])   ,(-1,1)) 
    day     = np.reshape(float(request.values['feature_day'])       ,(-1,1)) 
    month   = np.reshape(float(request.values['feature_month'])     ,(-1,1))
    duration= np.reshape(float(request.values['feature_duration'])  ,(-1,1))
    campaign= np.reshape(float(request.values['feature_campaign'])  ,(-1,1))
    pdays   = np.reshape(float(request.values['feature_pdays'])     ,(-1,1)) 
    previous= np.reshape(float(request.values['feature_previous'])  ,(-1,1)) 
    poutcome= np.reshape(float(request.values['feature_poutcome'])  ,(-1,1)) 

    feature_values = pd.DataFrame([[age, job, marital, edu, default, balance, house, loan, day,contact, month, duration, campaign, pdays, previous, poutcome]], columns=["age","job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome"], dtype=float, index=['input'])

    output = model.predict(feature_values)
    output = output.item 
    
    msg = 'success' if output == 1 else 'Failed' 

    return render_template('predictIon.html',prediction_text="The prediction is {}".format(msg))

    

if __name__ == '__main__':
    app.run(debug=True)

