import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageOps, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Load the Trained Model
model = tf.keras.models.load_model('mnist_model.h5')

# Define the Data Augmentation Function
def augment_image(image):
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1
    )

    image_array = np.array(image).reshape((1, 28, 28, 1))
    image_array = image_array / 255.0

    it = datagen.flow(image_array, batch_size=1)
    augmented_image = next(it)  # Use next(it) instead of it.next()

    augmented_image = augmented_image[0].reshape((28, 28)) * 255.0
    return Image.fromarray(augmented_image.astype('uint8'))


def open_file():
    file_path = filedialog.askopenfilename()

    if file_path:
        # Load and preprocess the image
        image = Image.open(file_path).convert('L')
        image = image.resize((28, 28))
        image = ImageOps.invert(image)

        # Resize for display purposes (increase the size, e.g., 100x100)
        display_image = image.resize((100, 100), Image.Resampling.LANCZOS)

        # Display the original image
        tk_image = ImageTk.PhotoImage(display_image)
        original_image_label.config(image=tk_image)
        original_image_label.image = tk_image

        # Augment the image
        augmented_image = augment_image(image)

        # Resize the augmented image for display
        display_aug_image = augmented_image.resize((100, 100), Image.Resampling.LANCZOS)

        # Display the augmented image
        tk_aug_image = ImageTk.PhotoImage(display_aug_image)
        augmented_image_label.config(image=tk_aug_image)
        augmented_image_label.image = tk_aug_image

        # Predict the digit
        pred_image = np.array(augmented_image).reshape((1, 28, 28, 1))
        prediction = model.predict(pred_image)
        predicted_digit = np.argmax(prediction)
        result_label.config(text=f"Predicted Digit: {predicted_digit}")


root = tk.Tk()
root.title("MNIST Digit Classifier")

root.geometry("600x800")

open_button = tk.Button(root, text='Open Image', command=open_file)
open_button.pack(pady=10)

original_image_label = tk.Label(root)
original_image_label.pack(pady=10)

augmented_image_label = tk.Label(root)
augmented_image_label.pack(pady=10)

result_label = tk.Label(root, text="")
result_label.pack(pady=10)

root.mainloop()


