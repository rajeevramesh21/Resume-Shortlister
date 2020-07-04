import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from werkzeug import secure_filename

app = Flask(__name__)
model = pickle.load(open('resumemodel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/getfile', methods=['GET','POST'])
def getfile():
    if request.method == 'POST':

        # for secure filenames. Read the documentation.
        file = request.files['myfile']
        filename = secure_filename(file.filename) 

        # os.path.join is used so that paths work in every operating system
        file.save(os.path.join("C:","Users","lenovo","Desktop","Resume Shortlister",filename))

        # You should use os.path.join here too.
        with open("C:/Users/lenovo/Desktop/Resume Shortlister") as f:
            file_content = f.read()
        features=[file_content]
        model.predict(features)
        prediction = model.predict(features)

        output = prediction

        if(output=='Data Science'):
            output1=' Resume has been shortlisted '
        else:
            output1='Resume not selected'

        return render_template('index.html', prediction_text=output1+ 'Resume Category is {}'.format(output))


    else:
        result = request.args.get['myfile']
    return result



if __name__ == "__main__":
    app.run(debug=True)