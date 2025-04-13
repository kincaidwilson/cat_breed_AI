import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image_dataset_from_directory
from PIL import Image
import numpy as np

# ========== STEP 0: Remove Unreadable/Corrupt Images ==========

def remove_corrupt_images(dataset_path):
    """Use PIL to remove obviously corrupt or unreadable images."""
    removed = 0
    for subdir, _, files in os.walk(dataset_path):
        for file in files:
            try:
                img_path = os.path.join(subdir, file)
                with Image.open(img_path) as img:
                    img.verify()
            except (IOError, SyntaxError):
                print(f"Removed corrupt image: {img_path}")
                os.remove(img_path)
                removed += 1
    print(f"PIL check done. Removed {removed} corrupt image(s).")

def remove_tf_unreadable_images(dataset_path):
    """Use TensorFlow’s JPEG decoder to catch any additional unreadable files."""
    removed = 0
    for subdir, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(subdir, file)
            try:
                img = tf.io.read_file(file_path)
                tf.image.decode_jpeg(img, channels=3)
            except Exception:
                print(f"Deleting unreadable image: {file_path}")
                os.remove(file_path)
                removed += 1
    print(f"TensorFlow check done. Removed {removed} unreadable image(s).")

# Run both checks
remove_corrupt_images("dataset")
remove_tf_unreadable_images("dataset")

# ========== STEP 1: Load and preprocess data ==========
data_dir = "dataset"
batch_size = 32
img_size = (224, 224)

train_ds = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

val_ds = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Detected classes:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# ========== STEP 2: Build model ==========
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ========== STEP 3: Train ==========
epochs = 10
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# ========== STEP 4: Save the model ==========
model.save("cat_breed_model.keras")

# ========== STEP 5: Plot results ==========
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Train Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend()
plt.title("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend()
plt.title("Loss")
plt.show()
