import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Sequential
import matplotlib.pyplot as plt
import os

# -----------------------------
# Configuration
# -----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "..",
    "dataset",
    "plantvillage dataset",
    "color"
)
IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

EPOCHS = 3

# -----------------------------
# Load Dataset
# -----------------------------

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# -----------------------------
# Class Names
# -----------------------------

class_names = train_dataset.class_names

print("\nDetected Classes:\n")

for index, name in enumerate(class_names):
    print(index, ":", name)

# -----------------------------
# Save Labels
# -----------------------------

with open("labels.txt", "w") as file:
    for label in class_names:
        file.write(label + "\n")

print("\nlabels.txt created successfully.")

# -----------------------------
# Optimize Dataset
# -----------------------------

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

# -----------------------------
# Normalize Images
# -----------------------------

normalization_layer = layers.Rescaling(1./255)

train_dataset = train_dataset.map(
    lambda x, y: (normalization_layer(x), y)
)

validation_dataset = validation_dataset.map(
    lambda x, y: (normalization_layer(x), y)
)

print("\nDataset Loaded Successfully!")

# -----------------------------
# Data Augmentation
# -----------------------------

data_augmentation = Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2)
])

## -----------------------------
# Build CNN Model
# -----------------------------

model = Sequential([

    layers.Input(shape=(224, 224, 3)),

    data_augmentation,

    layers.Conv2D(32, (3,3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(256, (3,3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(512, activation="relu"),

    layers.Dropout(0.5),

    layers.Dense(len(class_names), activation="softmax")

])
# -----------------------------
# Compile Model
# -----------------------------

model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)
model.build((None, 224, 224, 3))
# -----------------------------
# Model Summary
# -----------------------------

model.summary()

# -----------------------------
# Callbacks
# -----------------------------

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    verbose=1
)

# -----------------------------
# Train Model
# -----------------------------

history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=EPOCHS,

    callbacks=[
        checkpoint,
        early_stop,
        reduce_lr
    ]

)

# -----------------------------
# Save Final Model
# -----------------------------

model.save("model.h5")

print("\nModel saved successfully.")

# -----------------------------
# Plot Accuracy
# -----------------------------

plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"], label="Training")

plt.plot(history.history["val_accuracy"], label="Validation")

plt.title("Model Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid()

plt.show()

# -----------------------------
# Plot Loss
# -----------------------------

plt.figure(figsize=(8,5))

plt.plot(history.history["loss"], label="Training")

plt.plot(history.history["val_loss"], label="Validation")

plt.title("Model Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid()

plt.show()