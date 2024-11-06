import tensorflow as tf  # Import TensorFlow for machine learning tasks
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # For image augmentation and data loading
from tensorflow.keras import layers, models  # For creating and building the neural network
from tensorflow.keras.callbacks import ReduceLROnPlateau  # For adjusting the learning rate during training
from tensorflow.keras.applications import VGG16  # Import the pre-trained VGG16 model
import os  # For interacting with the file system

# Define paths to your numbers_dataset directory (adjust the path as necessary)
dataset_directory = "C:/Users/23058/PycharmProjects/UNO_CNN/numbers_dataset"  # Specify the path to the dataset

# Define image size (adjust based on your numbers_dataset)
image_size = (128, 128)  # Resize images to 128x128 pixels for consistency

# Define parameters for training
batch_size = 32  # Number of images processed per batch during training
epochs = 20  # Number of training epochs, you can increase this for better performance
learning_rate = 0.001  # Set the initial learning rate for the optimizer

# Create an ImageDataGenerator for loading and augmenting the numbers_dataset
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize the image pixel values to the range [0, 1]
    rotation_range=20,  # Randomly rotate images by up to 20 degrees
    width_shift_range=0.2,  # Randomly shift images horizontally by 20%
    height_shift_range=0.2,  # Randomly shift images vertically by 20%
    shear_range=0.2,  # Randomly shear images by 20%
    zoom_range=0.2,  # Randomly zoom in on images by 20%
    horizontal_flip=True,  # Randomly flip images horizontally
    fill_mode='nearest',  # Fill any newly created pixels with the nearest pixel value
    brightness_range=[0.8, 1.2],  # Randomly adjust the brightness of images within the specified range
    channel_shift_range=20.0  # Randomly shift the color channels by a certain range
)

# Create an ImageDataGenerator for validation (only rescaling)
val_datagen = ImageDataGenerator(rescale=1./255)  # Only rescale the validation data (no augmentation)

# Load the numbers_dataset from the directory and set up the training data generator
train_generator = train_datagen.flow_from_directory(
    os.path.join(dataset_directory, 'train'),  # Assuming your images are in the 'train' folder
    target_size=image_size,  # Resize images to the target size
    batch_size=batch_size,  # Set batch size for training
    class_mode='categorical',  # Each class corresponds to a unique number or type
)

# Set up the validation data generator
validation_generator = val_datagen.flow_from_directory(
    os.path.join(dataset_directory, 'val'),  # Assuming validation data is in the 'val' folder
    target_size=image_size,  # Resize validation images to the target size
    batch_size=batch_size,  # Set batch size for validation
    class_mode='categorical',  # Each class corresponds to a unique number or type
)

# Load the pre-trained VGG16 model as a base model, excluding the top layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(image_size[0], image_size[1], 3))
# We use the VGG16 model, without the fully connected layers, for feature extraction

# Freeze the layers of the base model to avoid training them
base_model.trainable = False  # The pre-trained VGG16 weights will not be updated during training

# Build the custom model by adding additional layers on top of the VGG16 base
model = models.Sequential([
    base_model,  # Add the pre-trained VGG16 model as the base
    layers.GlobalAveragePooling2D(),  # Perform global average pooling to reduce feature map size
    layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),  # Fully connected layer with L2 regularization
    layers.Dropout(0.5),  # Dropout layer to prevent overfitting (50% dropout rate)
    layers.Dense(len(train_generator.class_indices), activation='softmax')  # Output layer with a number of units equal to the number of classes
])

# Compile the model with Adam optimizer and categorical crossentropy loss function
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),  # Use Adam optimizer with the specified learning rate
    loss='categorical_crossentropy',  # Loss function for multi-class classification
    metrics=['accuracy']  # Track accuracy during training
)

# Print the model summary to verify the architecture
model.summary()  # Displays the model architecture to ensure it's correct

# Set up the learning rate scheduler to reduce the learning rate when validation loss plateaus
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)
# This will reduce the learning rate by a factor of 0.5 if the validation loss does not improve for 3 epochs

# Train the model
history = model.fit(
    train_generator,  # The training data generator
    steps_per_epoch=train_generator.samples // batch_size,  # Number of steps per epoch
    epochs=epochs,  # Number of epochs to train for
    validation_data=validation_generator,  # The validation data generator
    validation_steps=validation_generator.samples // batch_size,  # Number of validation steps
    callbacks=[lr_scheduler]  # Use the learning rate scheduler
)

# Save the trained model to a file for later use
model.save('uno_card_type_classifier_numbers.h5')  # Save the trained model as a .h5 file

# Set up a test data generator (if a 'test' folder is available)
test_generator = val_datagen.flow_from_directory(
    os.path.join(dataset_directory, 'test'),  # Assuming a 'test' folder is present
    target_size=image_size,  # Resize test images to the target size
    batch_size=batch_size,  # Set batch size for testing
    class_mode='categorical'  # Class mode for multi-class classification
)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
# Calculate the loss and accuracy of the model on the test data

# Print the test accuracy as a percentage
print(f"Test accuracy: {test_acc * 100:.2f}%")  # Print the accuracy of the model on the test dataset
