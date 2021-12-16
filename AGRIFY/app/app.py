# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 20:39:31 2021

@author: Keyur Chaniyara
"""

# from crop import predict
from flask import Flask, redirect, url_for,flash

app = Flask(__name__)
from flask import render_template, request
import pandas as pd
import pickle
import numpy as np
import joblib

yields = pd.read_csv("data Processed\clean_yield_prediction.csv")
modelyield = pickle.load(open("models\YieldDecisionTree.pkl","rb"))
state_model=joblib.load('models\State_le.joblib',mmap_mode = 'r')
District_model=joblib.load("models\District_le.joblib",mmap_mode = 'r')
Crop_model=joblib.load("models\Crop_le.joblib",mmap_mode = 'r')
Season_model=joblib.load('models\Season_le.joblib',mmap_mode = 'r')
crop = pd.read_csv("data Processed\crop_recommendation.csv")
modelcrop = pickle.load(open("models\CropRandomForest.pkl","rb"))
fert = pd.read_csv("data Processed\clean_fertilizer_prediction.csv")
modelfertilizer = pickle.load(open("models\FertRandomForest.pkl","rb"))


@app.route('/')
def index1():
    return render_template('index.html')

@app.route('/crops',methods=['GET'])
def index():
    return render_template('crop.html')
@app.route('/crops/predict',methods=['POST'])
def crops():
    if request.method == 'POST':
        nitrogen= request.form['nitrogen']
        phosphorus = request.form['phosphorus']
        potassium = request.form['potassium']
        temperature	= request.form['temperature']
        humidity = request.form['humidity']
        ph = request.form['ph']
        rainfall = request.form['rainfall']

        prediction = modelcrop.predict(pd.DataFrame([[nitrogen,phosphorus,potassium,temperature,humidity,ph,rainfall]],columns=['nitrogen','phosphours','potassium','temperature','humidity','ph','rainfall']))
        return str(prediction[0])
        # return render_template('crop.html',prediction_text="prediction : {}".format(output))
    else:
        return render_template('crop.html')

@app.route('/fertilizer',methods=['GET'])
def index2():
    ni = sorted(fert['Nitrogen'].unique())
    ph = sorted(fert['Phosphorus'].unique())
    po = sorted(fert['Potassium'].unique())
    cal	= sorted(fert['Calcium'].unique())
    mag = sorted(fert['Magnesium'].unique())
    sul = sorted(fert['Sulfur'].unique())
    li = sorted(fert['Lime'].unique())
    ca = sorted(fert['Carbon'].unique())
    mo = sorted(fert['Moisture'].unique())
    return render_template('fertilizer.html',ni = ni,ph=ph,po=po,cal=cal,mag=mag,sul=sul,li=li,ca=ca,mo=mo)
@app.route('/fertilizer/predict',methods=['POST'])
def fertilizer():
    if request.method == 'POST':
        nitrogen= request.form.get('nitrogen')
        phosphorus = request.form.get('phosphorus')
        potassium = request.form.get('potassium')
        calcium	= request.form.get('calcium')
        magnesium = request.form.get('magnesium')
        sulfur = request.form.get('sulfur')
        lime = request.form.get('lime')
        carbon = request.form.get('carbon')
        moisture = request.form.get('moisture')

        prediction = modelfertilizer.predict(pd.DataFrame([[nitrogen,phosphorus,potassium,calcium,magnesium,sulfur,lime,carbon,moisture]],columns=['nitrogen','phosphours','potassium','calcium','magnesium','sulfur','rainfall','carbon','moisture']))
        return str(prediction[0])
        # return render_template('fertilizer.html',prediction_text="prediction : {}".format(output))
    else:
        return render_template('fertilizer.html')

@app.route('/yield',methods=['GET'])
def index3():
    States = sorted(yields['State_Name'].unique())
    Districts = sorted(yields['District_Name'].unique()) 
    Years = sorted(yields['Crop_Year'].unique())
    Seasons	= sorted(yields['Season'].unique())
    Crops = sorted(yields['Crop'].unique())
    Areas = yields['Area']
    return render_template('yield.html',States = States,Districts=Districts,Years=Years,Seasons=Seasons,Crops=Crops,Areas=Areas)
@app.route('/yield/predict',methods=['POST'])
def Yield():
    if request.method == 'POST':
        State_Name= str(request.form.get('State_Name'))
        District_Name = str(request.form.get('District_Name'))
        Crop_Year = int(request.form.get('Crop_Year'))
        Season	= str(request.form.get('Season'))
        Crop = str(request.form.get('Crop'))
        Area = int(request.form.get('Area'))

        State_Name = state_model.transform([State_Name])
        District_Name = District_model.transform([District_Name]) 
        Season	= Season_model.transform([Season])     
        Crop = Crop_model.transform([Crop])

        prediction = modelyield.predict(pd.DataFrame([[State_Name,District_Name,Crop_Year,Season,Crop,Area]],columns=['State_Name','District_Name','Season','Crop_Year','Crop','Area']))
        return str(np.round(prediction[0],2))
    else:
        return render_template('yield.html')

if __name__=="__main__":
    app.run(debug=True)