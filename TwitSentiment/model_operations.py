import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Bidirectional, Dense, Dropout
from sklearn.metrics import accuracy_score, classification_report


def build_model(input_shape, num_layers=3, layer_size=50, layer_type='Dense', dropout_rate=0.2):
    """
    build a neural network model with the specified architecture.
    """
    model = Sequential()

    if layer_type == 'LSTM':
        model.add(LSTM(units=layer_size, return_sequences=(num_layers > 1), input_shape=input_shape))
    elif layer_type == 'GRU':
        model.add(GRU(units=layer_size, return_sequences=(num_layers > 1), input_shape=input_shape))
    elif layer_type == 'RNN':
        model.add(SimpleRNN(units=layer_size, return_sequences=(num_layers > 1), input_shape=input_shape))
    elif layer_type == 'BiGRU':
        model.add(Bidirectional(GRU(units=layer_size, return_sequences=(num_layers > 1), input_shape=input_shape)))
    elif layer_type == 'BiLSTM':
        model.add(Bidirectional(LSTM(units=layer_size, return_sequences=(num_layers > 1), input_shape=input_shape)))
    else:
        raise ValueError(f"Invalid layer type: {layer_type}")

    model.add(Dropout(dropout_rate))

    for _ in range(num_layers - 1):
        if layer_type == 'LSTM':
            model.add(LSTM(units=layer_size, return_sequences=(_ < num_layers - 1)))
        elif layer_type == 'GRU':
            model.add(GRU(units=layer_size, return_sequences=(_ < num_layers - 1)))
        elif layer_type == 'RNN':
            model.add(SimpleRNN(units=layer_size, return_sequences=(_ < num_layers - 1)))
        elif layer_type == 'BiGRU':
            model.add(Bidirectional(GRU(units=layer_size, return_sequences=(_ < num_layers - 1))))
        elif layer_type == 'BiLSTM':
            model.add(Bidirectional(LSTM(units=layer_size, return_sequences=(_ < num_layers - 1))))
        else:
            raise ValueError(f"Invalid layer type: {layer_type}")

        model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def train_model(model, X_train, y_train, epochs=25, batch_size=32):
    """
    train the model on the training data.
    """
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    return model


def evaluate_model(model, X_test, y_test):
    """
    evaluate the model on the test data.
    """
    predictions = model.predict(X_test) > 0.5.astype('int32')
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    return accuracy, report


def save_model(model, file_path):
    """
    save the model to a file.
    """
    model.save(file_path)
    joblib.dump(file_path, file_path + '.joblib')


def load_model(file_path):
    """
    load the model from a file.
    """
    model_path = joblib.load(file_path + '.joblib')
    model = load_model(model_path)
    return model

