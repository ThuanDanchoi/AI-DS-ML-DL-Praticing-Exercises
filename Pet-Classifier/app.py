from flask import Flask, request, render_template
import os
from predict import predict_image
from model import load_trained_model

# Load the pre-trained model
model = load_trained_model('my_pet_model.keras')

# Initialize Flask app
app = Flask(__name__)

# Ensure 'temp/' directory exists for saving uploaded files
if not os.path.exists('temp'):
    os.makedirs('temp')


@app.route("/", methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the POST request has the file part
        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']

        # If user does not select file, browser may submit an empty file without a filename
        if file.filename == '':
            return "No selected file"

        if file:
            # Save file to temp directory
            image_path = os.path.join('temp', file.filename)
            file.save(image_path)

            # Pass the uploaded image to the prediction function
            prediction = predict_image(model, image_path)
            return f"Prediction: {prediction}"

    return render_template('upload.html')


if __name__ == "__main__":
    app.run(debug=True)
