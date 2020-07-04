import numpy as np
from flask import Flask, request, jsonify, render_template, redirect
import pickle
from werkzeug import secure_filename

app = Flask(__name__)
model = pickle.load(open('resumemodel.pkl', 'rb'))

@app.route('/upload')
def upload_file():
   return render_template('index.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def predict():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      return 'file uploaded successfully'
	
if __name__ == "__main__":
    app.run(debug=True)