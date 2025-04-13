from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allows cross-origin requests (from React etc.)

# Load model
model = tf.keras.models.load_model("cat_breed_model.keras")

# Class labels (update as needed)
class_names = sorted([
    'Abyssinian', 'AmericanBobtail', 'AmericanCurl', 'AmericanShorthair',
    'AmericanWirehair', 'BalineseJavanese', 'Bengal', 'Birman', 'Bombay',
    'BritishShorthair', 'Burmese', 'Chartreux', 'CornishRex', 'DevonRex',
    'EgyptianMau', 'ExoticShorthair', 'HavanaBrown', 'Himalayan', 'JapaneseBobtail',
    'Korat', 'LaPerm', 'MaineCoon', 'Manx', 'Munchkin', 'NorwegianForestCat',
    'Ocicat', 'Persian', 'Pixiebob', 'Ragamuffin', 'Ragdoll', 'RussianBlue',
    'ScottishFold', 'Selkirk Rex', 'Siamese', 'Siberian', 'Singapura', 'Somali',
    'Sphynx', 'Tonkinese', 'TurkishAngora', 'TurkishVan'
])

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['file']
    try:
        img = Image.open(file.stream).resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_index = int(np.argmax(predictions))
        confidence = float(predictions[0][predicted_index])

        return jsonify({
            'breed': class_names[predicted_index],
            'confidence': round(confidence * 100, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
