from flask import Flask, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model
import numpy as np
import librosa
import os

app = Flask(__name__, static_folder='static')

# Memuat model
model = load_model('./musikemosi.h5')

def extract_features(audio_path, max_pad_len=174):
    try:
        audio, sr = librosa.load(audio_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        return mfccs
    except Exception as e:
        print(f"Error encountered while parsing file: {audio_path}, {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    file_path = os.path.join('./uploads', file.filename)
    file.save(file_path)

    features = extract_features(file_path)
    print(file_path)
    if features is None:
        print(f"Failed to extract features from {file_path}")
    else:
        features = features.reshape(1, -1)
    
    prediction = model.predict(features)
    # print(features.shape)
    # features = features.reshape(1, -1)
    # if features is None:
    #     raise ValueError("Features should not be None")
    # features = features.reshape(1, -1)

    # Melakukan prediksi
    # prediction = model.predict(features)
    predicted_class = np.argmax(prediction, axis=1)

    labels = ['Aggressive', 'Dramatic', 'Happy', 'Romantic', 'Sad']
    predicted_label = labels[predicted_class[0]]

    return jsonify({'prediction': predicted_label})

@app.route('/')
def index():
    return send_from_directory(app.static_folder, './index.html')

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
