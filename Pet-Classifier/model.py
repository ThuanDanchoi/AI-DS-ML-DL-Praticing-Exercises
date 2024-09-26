from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar100

# Define the pet classes we are interested in
PET_CLASSES = ['dog', 'cat', 'mouse', 'rabbit', 'bird']
PET_CLASS_INDICES = [5, 3, 71, 50, 2]  # Indices in CIFAR-100 for dog, cat, mouse, rabbit, bird

# Function to load and filter CIFAR-100 for selected pet classes
def load_filtered_cifar100():
    # Load CIFAR-100 dataset
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()

    # Filter only the selected classes
    train_filter = [i for i, label in enumerate(y_train) if label[0] in PET_CLASS_INDICES]
    test_filter = [i for i, label in enumerate(y_test) if label[0] in PET_CLASS_INDICES]

    x_train_filtered = x_train[train_filter]
    y_train_filtered = [PET_CLASS_INDICES.index(y[0]) for y in y_train[train_filter]]
    x_test_filtered = x_test[test_filter]
    y_test_filtered = [PET_CLASS_INDICES.index(y[0]) for y in y_test[test_filter]]

    # One-hot encode labels
    y_train_filtered = to_categorical(y_train_filtered, num_classes=len(PET_CLASSES))
    y_test_filtered = to_categorical(y_test_filtered, num_classes=len(PET_CLASSES))

    return (x_train_filtered, y_train_filtered), (x_test_filtered, y_test_filtered)

# Function to build and train the model
def train_model(save_model_path):
    # Load filtered data
    (x_train, y_train), (x_test, y_test) = load_filtered_cifar100()

    # Normalize image data
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Create a simple CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(len(PET_CLASSES), activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

    # Save the trained model
    model.save(save_model_path.replace('.h5', '.keras'))

def load_trained_model(model_path):
    return load_model(model_path)

 
