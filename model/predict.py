import os
import urllib.request

import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# MODEL PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "model.h5")

MODEL_URL = (
    "https://huggingface.co/MedhaviShenvi/"
    "agrivision-ai-model/resolve/main/model.h5"
)


# ============================================================
# DOWNLOAD MODEL IF NOT AVAILABLE
# ============================================================

if not os.path.exists(MODEL_PATH):

    print("Downloading model.h5...")

    urllib.request.urlretrieve(
        MODEL_URL,
        MODEL_PATH
    )

    print("Model download completed.")


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = tf.keras.models.load_model(MODEL_PATH)


# ============================================================
# LOAD CLASS LABELS
# ============================================================

LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")

with open(LABELS_PATH, "r") as file:
    class_names = [
        line.strip()
        for line in file.readlines()
    ]


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_disease(image_path):

    # Open image
    image = Image.open(image_path).convert("RGB")

    # Resize image
    image = image.resize((224, 224))

    # Convert image to array
    image_array = np.array(image)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # MobileNetV2 preprocessing
    image_array = preprocess_input(
        image_array.astype(np.float32)
    )

    # Prediction
    predictions = model.predict(
        image_array,
        verbose=0
    )

    # Get highest probability
    predicted_index = np.argmax(predictions[0])

    confidence = (
        predictions[0][predicted_index] * 100
    )

    disease = class_names[predicted_index]

    return disease, confidence


# ============================================================
# TEST PREDICTION
# ============================================================

if __name__ == "__main__":

    image_path = input("Enter image path: ")

    disease, confidence = predict_disease(
        image_path
    )

    print("\nPrediction:", disease)

    print(
        "Confidence: {:.2f}%".format(confidence)
    )