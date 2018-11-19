from datetime import datetime
import os
import time

from flask import Flask, render_template, jsonify, redirect, url_for, request
import skimage.io as skio


app = Flask(__name__)
app.config.from_object(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg', 'gif']

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            print(type(skio.imread(file)))
            return jsonify({"success":True, "grade": 2.45643})

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

if __name__ == "__main__":
    app.run(debug=True)
