import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from werkzeug import secure_filename



app = Flask(__name__)
model = pickle.load(open('resumemodel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    f = request.files['file']
    #f=request.form['file']
    data=[]
    with open (file,'r')as f:
        data=f.read()
    

    return render_template('index.html', prediction_text=data)
                           

if __name__ == "__main__":
    app.run(debug=True)