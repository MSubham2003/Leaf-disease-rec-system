import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image # type: ignore
from PIL import Image
import cv2
from keras.models import load_model # type: ignore
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
import requests

app = Flask(__name__)

# Google Drive Configuration
file_id = '1Vw0xr5rMU2T2AI8NP16c9MdZY_k3gKBl'  # Replace with your Google Drive file ID
download_url = f'https://drive.google.com/uc?id={file_id}'
local_model_path = 'model.h5'

def download_model_from_drive(url, local_path):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(local_path, 'wb') as f:
                f.write(response.content)
            print(f'{local_path} downloaded from Google Drive')
        else:
            print(f'Failed to download file, status code: {response.status_code}')
    except Exception as e:
        print(f'Error downloading file from Google Drive: {e}')

# Check if the model file already exists
if not os.path.exists(local_model_path):
    print('Model file not found. Downloading...')
    download_model_from_drive(download_url, local_model_path)
else:
    print('Model file already exists. Skipping download.')

# Load the model
model = load_model(local_model_path)
print('Model loaded. Check http://127.0.0.1:5000/')

labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}

def getResult(image_path):
    img = load_img(image_path, target_size=(225, 225))
    x = img_to_array(img)
    x = x.astype('float32') / 255.
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)[0]
    return predictions

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        # Ensure the 'uploads' folder exists
        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, 'uploads')
        if not os.path.exists(upload_path):
            os.makedirs(upload_path)

        file_path = os.path.join(upload_path, secure_filename(f.filename))
        f.save(file_path)

        predictions = getResult(file_path)
        predicted_label = labels[np.argmax(predictions)]
        return str(predicted_label)
    return None

if __name__ == '__main__':
    app.run(debug=True)
