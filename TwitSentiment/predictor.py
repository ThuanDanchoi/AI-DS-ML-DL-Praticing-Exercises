import numpy as np
import pandas as pd
from model_operations import load_model


def predict(model, data):
    """
    Predict the sentiment of the data.
    """
    predictions = model.predict(data) > 0.5.astype('int32')
    return predictions


def load_and_predict(model_path, data):
    model = load_model(model_path)
    data = pd.read_csv(data)

    predictions = predict(model, data)
    return predictions

