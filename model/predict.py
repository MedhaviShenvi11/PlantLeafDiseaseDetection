import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# Load trained model
model = tf.keras.models.load_model("model.h5")


# Load class labels
with open("labels.txt", "r") as file:
    class_names = [line.strip() for line in file.readlines()]


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
    image_array = preprocess_input(image_array.astype(np.float32))

    # Prediction
    predictions = model.predict(image_array, verbose=0)

    # Get highest probability
    predicted_index = np.argmax(predictions[0])

    confidence = predictions[0][predicted_index] * 100

    disease = class_names[predicted_index]

    return disease, confidence


# Test prediction
if __name__ == "__main__":

    image_path = input("Enter image path: ")

    disease, confidence = predict_disease(image_path)

    print("\nPrediction:", disease)
    print("Confidence: {:.2f}%".format(confidence))