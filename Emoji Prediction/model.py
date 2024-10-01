from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Build a simple LSTM model
model = Sequential([
    Embedding(input_dim=10000, output_dim=64, input_length=max_length),
    LSTM(64),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')  # Assuming 10 emoji categories
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(train_padded, train_labels, epochs=10, batch_size=32, validation_split=0.2)
