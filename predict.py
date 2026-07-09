import tensorflow as tf
import numpy as np

class_names = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck"
]

model = tf.keras.models.load_model("saved_model/cnn_model.keras")

def predict_image(image):
    # Convert RGBA or grayscale to RGB
    image = image.convert("RGB")

    # Resize to model input size
    image = image.resize((32, 32))

    # Convert to NumPy array
    image = np.array(image).astype("float32") / 255.0

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    # Predict
    prediction = model.predict(image, verbose=0)

    predicted_class = np.argmax(prediction)

    return class_names[predicted_class]
