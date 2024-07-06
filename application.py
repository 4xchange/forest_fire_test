import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template




application=Flask(__name__)
app=application

model=pickle.load(open('models/model.pkl','rb'))
scaler=pickle.load(open('models/scaler.pkl','rb'))

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata",methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        temp=float(request.form.get('Temperature'))
        rh=float(request.form.get('RH'))
        ws=float(request.form.get('Ws'))
        rain=float(request.form.get('Rain'))
        ffmc=float(request.form.get('FFMC'))
        dmc=float(request.form.get('DMC'))
        isi=float(request.form.get('ISI'))
        classes=float(request.form.get('Classes'))
        region=float(request.form.get('Region'))


        scaled_data=scaler.transform([[temp,rh,ws,rain,ffmc,dmc,isi,classes,region]])
        result=model.predict(scaled_data)

        return render_template("home.html",result=result[0])

    else:
        return render_template("home.html")


if __name__=="__main__":
    app.run(host="0.0.0.0")