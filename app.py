from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load('model.pkl','rb')

@app.route('/', methods=["GET","POST"])
def home():
    if request.method=="GET":
        return render_template('index.html')
    else:
        age     = np.reshape(float(request.values['feature_age'])       ,(-1,1))
        job     = np.reshape(float(request.values['feature_job'])       ,(-1,1))
        marital = np.reshape(float(request.values['feature_marital'])   ,(-1,1))
        edu     = np.reshape(float(request.values['feature_education']) ,(-1,1))
        default = np.reshape(float(request.values['feature_default'])   ,(-1,1)) 
        balance = np.reshape(float(request.values['feature_balance'])   ,(-1,1))
        house   = np.reshape(float(request.values['feature_housing'])   ,(-1,1))
        loan    = np.reshape(float(request.values['feature_loan'])      ,(-1,1))
        contact = np.reshape(float(request.values['feature_contact'])   ,(-1,1)) 
        contact = np.reshape(float(request.values['feature_day'])       ,(-1,1)) 
        month   = np.reshape(float(request.values['feature_month'])     ,(-1,1))
        duration= np.reshape(float(request.values['feature_duration'])  ,(-1,1))
        campaign= np.reshape(float(request.values['feature_campaign'])  ,(-1,1))
        pdays   = np.reshape(float(request.values['feature_pdays'])     ,(-1,1)) 
        previous= np.reshape(float(request.values['feature_previous'])  ,(-1,1)) 
        poutcome= np.reshape(float(request.values['feature_poutcome'])  ,(-1,1)) 

        output = model.predict(age, job, marital, edu, default, balance, house, loan, contact, month, duration, campaign, pdays, previous, poutcome)
        output = output.item
        output = round(output,2)

        msg = 'success' if output == 1 else 'Failed' 

        return render_template('index.html', prediction_text='The prediction is '.format(msg))

@app.route('/predict',methods =['POST'])
def predict():
    feature=[float(feature)for feature in request.form.values()]
    feature_list= [np.array(feature)]
    output=model.predict(feature_list)
    return render_template('prediciton.html',prediction_text="The prediction is {}".format(output))

    

if __name__ == '__main__':
    app.run(debug=True)


