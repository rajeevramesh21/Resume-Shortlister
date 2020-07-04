import numpy as np
from flask import Flask, request, jsonify, render_template, redirect
import pickle
from werkzeug import secure_filename

app = Flask(__name__)
model = pickle.load(open('resumemodel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')
	
@app.route('/predict',methods=['POST'])
def predict():
   if request.method == 'POST':
      f = request.files['file']
      #f.save(secure_filename(f.filename))
      #with open ('file.txt') as fo:
          #for rec in fo:
              #data=rec
      return render_template('index.html', 'Resume Category is {}')
  return render_template('index.html', 'Resume Category is {}')
if __name__ == "__main__":
    app.run(debug=True)