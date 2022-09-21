import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import pickle
import numpy as np

st.title("Employee Promotion Project")

st.info("Select values for each of the following features")

dept = st.selectbox("Department",('Sales & Marketing','Operations','Technology','Procurement','Analytics',
                                  'Finance','HR','Legal','R&D'))
region = st.selectbox("Region",('1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18',
                                '19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34'))
edu = st.selectbox("Education",("Bachelors","Masters & above","Below Secondary"))
gender = st.selectbox("Gender",("Male","Female"))
ch = st.selectbox("Recruitment channel",("referred","sourcing","other"))
no_of_train = st.selectbox("Number of training",('1','2','3','4','5','6','7','8','9','10'))
age = st.number_input("Enter age",min_value=20,max_value=60)
previous_rating = st.selectbox("Previous year rating",('1','2','3','4','5'))
service = st.number_input("Enter length of service",min_value=1,max_value=37)
award_won = st.selectbox("Award won in previous year",('Yes','No'))
train_score = st.number_input("Enter average training score",min_value=39,max_value=99)

X = np.zeros((1, 55))
if edu == 'Below Secondary':
    X[0][0] = 1
elif edu == 'Bachelors':
    X[0][0] = 2
elif edu == 'Masters & above':
    X[0][0] = 3

X[0][1] = float(no_of_train)
X[0][2] = age
X[0][3] = float(previous_rating)
X[0][4] = service

if award_won == 'Yes':
    X[0][5] = 1
    
X[0][6] = train_score

if dept == 'Analytics':
    X[0][7] = 1
elif dept == 'Finance':
    X[0][8] = 1
elif dept == 'HR':
    X[0][9] = 1
elif dept == 'Legal':
    X[0][10] = 1
elif dept == 'Operations':
    X[0][11] = 1
elif dept == 'Procurement':
    X[0][12] = 1
elif dept == 'R&D':
    X[0][13] = 1
elif dept == 'Sales & Marketing':
    X[0][14] = 1
elif dept == 'Technology':
    X[0][15] = 1

region = int(region)
if region == 1:
    X[0][16] = 1
elif region in range(10,20,1):
    X[0][region+7] = 1   
elif region == 2:
    X[0][27] = 1       
elif region in range(20,30,1):
    X[0][region+8] = 1   
elif region == 3:
    X[0][38] = 1
elif region in range(30,35,1):
    X[0][region+9] = 1
elif region in range(4,10,1):
    X[0][region+40] = 1
    
if gender == 'Female':
    X[0][50] = 1
elif gender == 'Male':
    X[0][51] = 1
    
if ch == 'other':    
    X[0][52] = 1
elif ch == 'referred':    
    X[0][53] = 1
elif ch == 'sourcing':    
    X[0][54] = 1
    
with open('P151_scaler' , 'rb') as sc:
    scaler = pickle.load(sc)

X_scaled = scaler.transform(X)

with open('P151_model' , 'rb') as file:
    model = pickle.load(file)
    
y = model.predict(X_scaled)

if y == 1:
    st.success("Employee may be promoted.")
else:
    st.warning("Employee may not be promoted.") 
