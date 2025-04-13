import tensorflow as tf
import numpy as np
from PIL import Image

# Load the new .keras model (saved from your training script)
model = tf.keras.models.load_model("cat_breed_model.keras")

# Replace this list with your actual class names from dataset folders
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

# Load and preprocess test image
img_path = "test_cat.jpg"  # Replace with your test image file
img = Image.open(img_path).resize((224, 224))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Make prediction
predictions = model.predict(img_array)
predicted_index = np.argmax(predictions)
predicted_class = class_names[predicted_index]
confidence = predictions[0][predicted_index] * 100

print(f"Predicted Breed: {predicted_class} ({confidence:.2f}%)")
