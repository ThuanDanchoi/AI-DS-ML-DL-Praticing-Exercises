from taipy.gui import Gui
from tensorflow.keras import models
from PIL import Image
import numpy as np

class_names = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck',
}

# Attempt to load the model
try:
    model = models.load_model("baseline.keras")  # Correct model filename
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def predict_image(model, path_to_img):
    try:
        img = Image.open(path_to_img)
        img = img.convert("RGB")
        img = img.resize((32, 32))
        data = np.asarray(img)
        data = data / 255
        probs = model.predict(np.array([data])[:1])
        top_prob = probs.max()
        top_pred = class_names[np.argmax(probs)]
        return top_prob, top_pred
    except Exception as e:
        print(f"Error predicting image: {e}")
        return 0, "unknown"

content = ""
img_path = "placeholder_image.png"
prob = 0
pred = ""

index = """
<|text-center|
<|{"logo.png"}|image|width=25vw|>

<|{content}|file_selector|extensions=.png|>
select an image from your file system

<|{pred}|>

<|{img_path}|image|>

<|{prob}|indicator|value={prob}|min=0|max=100|width=25vw|>
>
"""

def on_change(state, var_name, var_val):
    global model
    if var_name == "content":
        if model is not None:
            top_prob, top_pred = predict_image(model, var_val)
            state.prob = round(top_prob * 100)
            state.pred = "this is a " + top_pred
            state.img_path = var_val
        else:
            state.pred = "Model not loaded."
            state.prob = 0
            state.img_path = var_val

app = Gui(page=index)

if __name__ == "__main__":
    app.run(use_reloader=True, port=5001)

