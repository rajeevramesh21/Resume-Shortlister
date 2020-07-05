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
    f=request.files['file']

    

    return render_template('index.html', prediction_text='Resume Shortlisted')
                           

if __name__ == "__main__":
    app.run(debug=True)