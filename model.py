import pandas as pd 
from sklearn.model_selection import train_test_split 

import pickle
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler

df = pd.read_excel(r"/main_project-bank-full.xlsx")

# Handling missing values
df1=df.fillna(method='pad')

# encoding
df1['job'] = df1['job'].replace({'management' :1, 'technician' :2, 'entrepreneur':3,'blue-collar' :4, 'retired' :5, 'admin.':6, 'services' :7, 'self-employed' :8, 'unemployed' :9, 'housemaid':10, 'student':11, 'unknown':-1 })
df1['education'] = df1['education'].replace({'tertiary' :1, 'secondary' :2, 'primary':3, 'unknown':-1})
df1['contact'] = df1['contact'].replace({'cellular' :1, 'telephone' :2, 'unknown':-1})
df1['month'] = df1['month'].replace({'jan' :1, 'feb' :2, 'mar':3, 'apr' :4, 'may' :5, 'jun':6, 'jul' :7, 'aug' :8, 'sep' :9, 'oct':10, 'nov':11, 'dec':12 })
df1['poutcome'] = df1['poutcome'].replace({'failure' :1, 'other' :2,'success' :3, 'unknown':-1})

# Balancing DataSet via SMOTE
sm = SMOTE()
X_sm, y_sm = sm.fit_resample(X, y)


# Scaling
X = df1.drop(['Target'], axis=1)
y = df1['Target']
# split into 70:30 ration
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_sm, y_sm, test_size = 0.3, random_state = 0)

# define the pipeline 
trans = MinMaxScaler()
min_max = MinMaxScaler(feature_range= (0,1))
X_train2 = min_max.fit_transform(X_train1)
X_train2 = pd.DataFrame(X_train2) 


# what is model
# where is regression/regressor ?


pickle.dump(regressor,open('model.pkl','wb'))

# run this file to create model.pkl