from flask import Flask, request, request, jsonify, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads/'
model = load_model('cat_rabbit_classification.keras') #Load model

labels = ['Cat', 'Rabbit'] #List label
predictions = []  # List untuk menyimpan hasil prediksi

def predict_label(img_path):
    img = image.load_img(img_path, target_size=(128, 128)) #Menyesuaikan dengan ukuran input model
    if img is None:
        print("Gambar tidak dapat dibaca. Periksa path gambar")
        return None
    
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan dimensi batch
    img_array /= 255.0  # Normalisasi

    prediction = model.predict(img_array) #melakukan prediksi
    #predicted_label = np.argmax(prediction)
    if prediction[0]>0.5:
        predicted_label=1
    else:
        predicted_label = 0
    return predicted_label, labels[predicted_label]

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    # Simpan gambar yang diunggah
    uploaded_image = request.files['image']
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_image.filename)
    uploaded_image.save(img_path)
    
    predicted_label, predicted_label_name = predict_label(img_path)
    
    if predicted_label_name is None:
        return jsonify({'error': 'Failed to predict image'}), 400
    # Buat URL untuk mengakses gambar yang diunggah
    image_url = url_for('static', filename=f'uploads/{uploaded_image.filename}', _external=True)

    prediction_data = {
            'image_url': image_url,
            'predicted_label': int(predicted_label),
            'label_name': predicted_label_name,
            }
    predictions.append(prediction_data)  # Tambah ke riwayat prediksi

    return jsonify(prediction_data)

@app.route('/predictions', methods=['GET'])
def get_predictions():
    # Mengembalikan seluruh hasil prediksi yang disimpan dalam list
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
