from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

def load_and_prepare_image(img_path, img_size=(150, 150)):
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict(model_path='camera_classifier_model.h5', img_path='test_image.jpg'):
    model = load_model(model_path)
    img_array = load_and_prepare_image(img_path)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    return predicted_class, confidence

if __name__ == "__main__":
    img_path = input("Enter the path to the camera image: ")
    predicted_class, confidence = predict(img_path=img_path)
    print(f"Predicted class: {predicted_class} with confidence: {confidence}")
