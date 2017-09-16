import os
import keras
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
from keras.models import load_model
import cv2
import numpy as np
import time

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
model = load_model('model.h5')
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['POST'])

def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            time.sleep(1)

            img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = img.reshape(1, 224, 224, 3)

            img = (img - np.mean(img))/np.std(img)

            return str(np.amax(model.predict(img)))

    return
