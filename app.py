import numpy as np
from flask import Flask, request, jsonify, render_template, redirect
import pickle
from werkzeug import secure_filename

app = Flask(__name__)
model = pickle.load(open('resumemodel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':

        f = request.form['content']
        data=[]
        with open(f) as file:
            data=file.read()
        return data,f
if __name__ == "__main__":
    app.run(debug=True)