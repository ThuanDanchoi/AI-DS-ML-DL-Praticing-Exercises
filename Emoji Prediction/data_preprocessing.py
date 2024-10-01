import tensorflow as tf
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Load Train and Test CSV files
train_df = pd.read_csv('data/Train.csv')
test_df = pd.read_csv('data/Test.csv')

# Split the Train data into X (TEXT) and Y (Label)
X_train = train_df['TEXT']
Y_train = train_df['Label']

X_test = test_df['TEXT']

# Apply Random Oversampling to balance classes in the training set
os = RandomOverSampler()
X_train_resampled, Y_train_resampled = os.fit_resample(X_train.values.reshape(-1, 1), Y_train)

X_train_resampled = X_train_resampled.flatten()

# Tokenize and pad the text data
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_resampled)

# Convert texts to sequences of numbers
train_sequences = tokenizer.texts_to_sequences(X_train_resampled)
test_sequences = tokenizer.texts_to_sequences(X_test)

# Pad the sequences to ensure they all have the same length
max_length = 100  # You can adjust this based on your dataset
X_train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
X_test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

# Check the unique values in Y_train_resampled to identify the label range
unique_labels = set(Y_train_resampled)
print("Unique labels in Y_train_resampled:", unique_labels)

# Find the maximum label value in Y_train_resampled
num_classes = max(Y_train_resampled) + 1  # Ensure num_classes equals the highest label + 1

# Convert labels to categorical (for emoji classification)
train_labels = tf.keras.utils.to_categorical(Y_train_resampled, num_classes=num_classes)

# Check if the issue is resolved
print("Shape of train_labels:", train_labels.shape)

X_train_final, X_val_final, Y_train_final, Y_val_final = train_test_split(
    X_train_padded, train_labels, test_size=0.2, random_state=42
)

# Check the shape of the splits
print("Training data shape:", X_train_final.shape)
print("Validation data shape:", X_val_final.shape)
print("Training labels shape:", Y_train_final.shape)
print("Validation labels shape:", Y_val_final.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Define the model
model = Sequential([
    Embedding(input_dim=10000, output_dim=64, input_length=100),  # Adjust dimensions
    LSTM(64, return_sequences=False),  # LSTM layer with 64 units
    Dense(32, activation='relu'),  # Fully connected layer
    Dense(20, activation='softmax')  # Output layer for 20 emoji classes
])

# Build the model and print summary
model.build(input_shape=(None, 100))
model.summary()

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train_final, Y_train_final,  # Use already one-hot encoded labels
    epochs=10,
    batch_size=32,
    validation_data=(X_val_final, Y_val_final)  # Use one-hot encoded labels for validation
)

# Evaluate the model
model.evaluate(X_test_padded, test_df['Label'], batch_size=32)

# Plot loss and accuracy
loss = pd.DataFrame(history.history)
print(loss)
