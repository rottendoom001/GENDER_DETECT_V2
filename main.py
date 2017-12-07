
import scipy.io.wavfile
from flask import Flask, request, redirect, jsonify, render_template
from gender_detector import GenderDetector

ALLOWED_EXTENSIONS = set(['wav'])

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and filename.rsplit(
        '.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file_wav = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file_wav.filename == '':
            return redirect(request.url)
        if file_wav and allowed_file(file_wav.filename):
            fs, signal = scipy.io.wavfile.read(file_wav)
            result = GenderDetector().process(fs, signal)
            return jsonify(result)

    return render_template('main.html')
