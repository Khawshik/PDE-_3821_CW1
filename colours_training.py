import tensorflow as tf  # Importing the TensorFlow library for building and training models
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Importing ImageDataGenerator for loading and augmenting images
from tensorflow.keras import layers, models  # Importing layers and models from Keras to define the neural network
import os  # Importing os for handling file paths and directories

# Path to dataset directory
dataset_directory = "C:/Users/23058/PycharmProjects/UNO_CNN/numbers_dataset"  # Change this path

# Image size (adjust based on your dataset)
image_size = (128, 128)  # Resize images to 128x128

# Define parameters for training
batch_size = 32  # Number of images processed together in one step
epochs = 50  # Number of times the entire dataset will be passed through the model (adjust for better performance)
learning_rate = 0.001  # Learning rate for the Adam optimizer

# Create an ImageDataGenerator for loading and augmenting the dataset
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0, 1]
    rotation_range=30,  # Random rotations
    width_shift_range=0.2,  # Horizontal shift
    height_shift_range=0.2,  # Vertical shift
    shear_range=0.2,  # Shear transformations
    zoom_range=0.2,  # Random zoom
    horizontal_flip=True,  # Flip images horizontally
    fill_mode='nearest'  # Filling missing pixels
)

# Create an ImageDataGenerator for validation (only rescaling)
val_datagen = ImageDataGenerator(rescale=1./255)  # Only normalize the pixel values for validation data

# Load the dataset from the directory
train_generator = train_datagen.flow_from_directory(
    os.path.join(dataset_directory, 'train'),  # Path to the 'train' folder in the dataset directory
    target_size=image_size,  # Resize all images to the defined image size
    batch_size=batch_size,  # Use the defined batch size
    class_mode='categorical',  # Use categorical encoding for multi-class classification
)

validation_generator = val_datagen.flow_from_directory(
    os.path.join(dataset_directory, 'validation'),  # Path to the 'validation' folder in the dataset directory
    target_size=image_size,  # Resize images for validation
    batch_size=batch_size,  # Use the same batch size as training
    class_mode='categorical',  # Use categorical encoding for validation data
)

# A simple CNN model
model = tf.keras.models.Sequential([  # Sequential model to define the layers of the neural network
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)),  # First convolutional layer with 32 filters
    tf.keras.layers.MaxPooling2D((2, 2)),  # Max pooling layer to reduce the spatial dimensions
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),  # Second convolutional layer with 64 filters
    tf.keras.layers.MaxPooling2D((2, 2)),  # Max pooling layer
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),  # Third convolutional layer with 128 filters
    tf.keras.layers.MaxPooling2D((2, 2)),  # Max pooling layer
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),  # Fourth convolutional layer with 256 filters
    tf.keras.layers.MaxPooling2D((2, 2)),  # Max pooling layer
    tf.keras.layers.Flatten(),  # Flatten the output for the fully connected layers
    tf.keras.layers.Dense(256, activation='relu'),  # Dense layer with 256 units
    tf.keras.layers.Dropout(0.5),  # Dropout layer to prevent overfitting
    tf.keras.layers.Dense(len(train_generator.class_indices), activation='softmax')  # Output layer with the number of classes in the dataset
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),  # Adam optimizer with the specified learning rate
    loss='categorical_crossentropy',  # Categorical cross-entropy loss for multi-class classification
    metrics=['accuracy']  # Track accuracy during training
)

# Train the model
history = model.fit(
    train_generator,  # Training data generator
    steps_per_epoch=train_generator.samples // batch_size,  # Steps per epoch based on batch size and number of samples
    epochs=epochs,  # Number of epochs to train the model
    validation_data=validation_generator,  # Validation data generator
    validation_steps=validation_generator.samples // batch_size  # Steps per validation epoch
)

# Save the model
model.save('uno_card_classifier_colours.h5')  # Save the trained model to a file

# Evaluate the model (optional)
test_generator = val_datagen.flow_from_directory(
    os.path.join(dataset_directory, 'test'),  # Path to the 'test' folder in the dataset directory
    target_size=image_size,  # Resize images for testing
    batch_size=batch_size,  # Use the same batch size as training
    class_mode='categorical',  # Categorical encoding for test data
)

# Evaluate performance on test data
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // batch_size)  # Evaluate the model on the test data
print(f"Test accuracy: {test_acc * 100:.2f}%")  # Print the test accuracy as a percentage
