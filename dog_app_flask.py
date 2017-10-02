import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from flask import send_from_directory
from dog_recognizer import predict_breed

from flask import send_from_directory




UPLOAD_FOLDER = 'uploads/dog_photos'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/dog_recognizer/dog_photos/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
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
            return redirect(url_for('dog_recognizer',filename=filename))
    return render_template('upload.html')

@app.route('/dog_recognizer/<filename>')
def dog_recognizer(filename):
    path_to_photo = 'uploads/dog_photos/'+filename
    race, breed = predict_breed(path_to_photo)
    uploaded_file(filename)
    if race == 'neither':
        return render_template('neither.html',path='dog_photos/'+filename)
    elif race == 'dog':
        return render_template('dog.html',breed=breed, path='dog_photos/'+filename)
    else:
        return render_template('human.html',breed=breed, path='dog_photos/'+filename)




if __name__ == '__main__':
   app.run(debug = True)