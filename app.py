"""
Created on oct 15 2021, 17:34:25

created by - Babita Pant
"""


from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('Model.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['Time']
    data2 = request.form['v1']
    data3 = request.form['v2']
    data4 = request.form['v3']
    data5 = request.form['v4']
    data6 = request.form['v5']
    data7 = request.form['v6']
    data8 = request.form['v7']
    data9 = request.form['v8']
    data10 = request.form['v9']
    data11 = request.form['v10']
    data12 = request.form['v11']
    data13= request.form['v12']
    data14= request.form['v13']
    data15= request.form['v14']
    data16= request.form['v15']
    data17= request.form['v16']
    data18= request.form['v17']
    data19= request.form['v18']
    data20= request.form['v19']
    data21= request.form['v20']
    data22= request.form['v21']
    data23= request.form['v22']
    data24= request.form['v23']
    data25 = request.form['v24']
    data26= request.form['v25']
    data27 = request.form['v26']
    data28= request.form['v27']
    data29= request.form['v28']
    data30= request.form['Amount']





    arr = np.array([[data1, data2, data3, data4, data5, data6, data7, data8, data9,data10, data11, data12, data13, data14,
                     data15, data16, data17,
                     data18, data19,data20,data21,data22, data23, data24, data25, data26, data27,data28,data29,data30]])
    pred = model.predict(arr)
    return render_template('result.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)