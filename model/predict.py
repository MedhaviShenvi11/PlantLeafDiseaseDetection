import os

import numpy as np
from PIL import Image
import tensorflow as tf


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model.tflite"
)

LABELS_PATH = os.path.join(
    BASE_DIR,
    "labels.txt"
)


# ============================================================
# LOAD TFLITE MODEL
# ============================================================

interpreter = tf.lite.Interpreter(
    model_path=MODEL_PATH
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# ============================================================
# LOAD CLASS LABELS
# ============================================================

with open(LABELS_PATH, "r") as file:

    class_names = [
        line.strip()
        for line in file.readlines()
    ]


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_disease(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image = image.resize(
        (224, 224)
    )

    image_array = np.array(
        image
    ).astype(
        np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # MobileNetV2 preprocessing
    image_array = (
        image_array / 127.5
    ) - 1.0

    interpreter.set_tensor(
        input_details[0]["index"],
        image_array
    )

    interpreter.invoke()

    predictions = interpreter.get_tensor(
        output_details[0]["index"]
    )

    predicted_index = np.argmax(
        predictions[0]
    )

    confidence = (
        predictions[0][predicted_index] * 100
    )

    disease = class_names[
        predicted_index
    ]

    return disease, confidence