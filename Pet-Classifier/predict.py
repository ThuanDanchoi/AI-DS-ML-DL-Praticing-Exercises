from tensorflow.keras.preprocessing import image
import numpy as np

# Define the pet classes
PET_CLASSES = ['dog', 'cat', 'mouse', 'rabbit', 'bird']

# Function to predict an image using the pre-trained model
def predict_image(model, image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(32, 32))  # CIFAR-100 image size is 32x32
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize

    # Make the prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)  # Get the class with the highest probability

    return PET_CLASSES[predicted_class[0]]
