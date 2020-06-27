import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

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
    int_features = [str(x) for x in request.form.values()]
    #final_features = [int_features]
    prediction = model.predict(int_features)

    output = prediction

    return render_template('index.html', prediction_text='Resume Category is {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)