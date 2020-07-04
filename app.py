import numpy as np
from flask import Flask, request, jsonify, render_template
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

        # for secure filenames. Read the documentation.
        f = request.form['content']
        data=[]
        with open(f) as file:
            data=file.read()
        
        model.predict(data)
        prediction = model.predict(data)

        output = prediction

        if(output=='Data Science'):
            output1=' Resume has been shortlisted '
        else:
            output1='Resume not selected'

        return render_template('index.html', prediction_text=output1+ 'Resume Category is {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)