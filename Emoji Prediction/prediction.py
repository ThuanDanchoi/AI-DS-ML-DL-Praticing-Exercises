import tensorflow as tf
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
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
train_labels = tf.keras.utils.to_categorical(Y_train_resampled, num_classes=20)

# Check if the issue is resolved
print("Shape of train_labels:", train_labels.shape)

# Plot original label distribution before oversampling
plt.figure(figsize=(10, 6))
sns.countplot(x=Y_train)
plt.title('Original Label Distribution Before Oversampling')
plt.show()

# Plot label distribution after oversampling
plt.figure(figsize=(10, 6))
sns.countplot(x=Y_train_resampled)
plt.title('Label Distribution After Oversampling')
plt.show()