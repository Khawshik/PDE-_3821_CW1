# Import necessary libraries
import cv2  # OpenCV for image processing and computer vision tasks
import tensorflow as tf  # TensorFlow for loading and using the trained models
import numpy as np  # NumPy for numerical operations and array handling

# Load the color and number models (these are pre-trained models for detecting card colors and numbers)
color_model = tf.keras.models.load_model('C:/Users/23058/PycharmProjects/UNO_CNN/uno_card_classifier_colours.h5')  # Path to color model
number_model = tf.keras.models.load_model('C:/Users/23058/PycharmProjects/UNO_CNN/uno_card_type_classifier_numbers.h5')  # Path to number model

# Define class names for color and number predictions
color_class_names = ['red', 'yellow', 'green', 'blue']  # Color classes the model will predict
number_class_names = [str(i) for i in range(10)]  # Number classes from 0 to 9 for UNO cards

# Function to preprocess an image to match the model's input format
def preprocess_image(image):
    img = cv2.resize(image, (128, 128))  # Resize image to 128x128 (adjust size based on model requirements)
    img = img.astype("float32") / 255.0  # Normalize pixel values to range [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension (models expect input shape to have 4 dimensions)
    return img

# Function to detect and classify a UNO card in a given image file
def detect_card_from_image(image_path):
    image = cv2.imread(image_path)  # Read the image from the specified file path
    if image is None:  # Check if image is loaded correctly
        print("Could not read image from file.")
        return

    preprocessed_image = preprocess_image(image)  # Preprocess the image for model input

    # Predict the color and number of the card separately using the models
    color_predictions = color_model.predict(preprocessed_image)
    number_predictions = number_model.predict(preprocessed_image)

    color_index = np.argmax(color_predictions)  # Get the index of the highest prediction for color
    color_confidence = np.max(color_predictions)  # Get the confidence score for the color prediction
    number_index = np.argmax(number_predictions)  # Get the index of the highest prediction for number
    number_confidence = np.max(number_predictions)  # Get the confidence score for the number prediction

    # Get the labels (color and number) based on the predicted index
    color_label = color_class_names[color_index]
    number_label = number_class_names[number_index]

    # Display the prediction if confidence is above the threshold
    confidence_threshold = 0.5  # Set the confidence threshold for valid predictions
    if color_confidence >= confidence_threshold and number_confidence >= confidence_threshold:
        # Annotate the image with the predicted color and number along with their confidence scores
        cv2.putText(image, f'{color_label} {number_label} ({color_confidence:.2f}, {number_confidence:.2f})',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("UNO Card Detection", image)  # Display the annotated image
        cv2.waitKey(0)  # Wait for a key press to close the window
        cv2.destroyAllWindows()  # Close the OpenCV window after key press
    else:
        print(f"Prediction below confidence threshold: {color_label} {number_label}")

# Function to detect and classify a UNO card from a live camera feed
def detect_card_from_camera():
    cap = cv2.VideoCapture(0)  # Open the default camera (camera index 0)
    if not cap.isOpened():  # Check if the camera opened successfully
        print("Could not open camera.")
        return

    try:
        while True:
            ret, frame = cap.read()  # Capture a frame from the camera
            if not ret:  # Check if the frame was captured successfully
                print("Failed to capture image.")
                break

            preprocessed_frame = preprocess_image(frame)  # Preprocess the captured frame for model input

            # Predict the color and number of the card separately using the models
            color_predictions = color_model.predict(preprocessed_frame)
            number_predictions = number_model.predict(preprocessed_frame)

            color_index = np.argmax(color_predictions)  # Get the index of the highest prediction for color
            color_confidence = np.max(color_predictions)  # Get the confidence score for the color prediction
            number_index = np.argmax(number_predictions)  # Get the index of the highest prediction for number
            number_confidence = np.max(number_predictions)  # Get the confidence score for the number prediction

            color_label = color_class_names[color_index]  # Get the predicted color label
            number_label = number_class_names[number_index]  # Get the predicted number label

            # If both predictions exceed the confidence threshold, annotate and display the frame
            if color_confidence >= 0.5 and number_confidence >= 0.5:
                cv2.putText(frame, f'{color_label} {number_label} ({color_confidence:.2f}, {number_confidence:.2f})',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("UNO Card Detection", frame)  # Display the annotated frame

            # Print the detected result and confidence scores to the console
            print(f"Detected: {color_label} {number_label} with confidence ({color_confidence:.2f}, {number_confidence:.2f})")

            # Exit the loop if the 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()  # Release the camera when done
        cv2.destroyAllWindows()  # Close any OpenCV windows

# Main function to prompt user for detection mode and call the respective function
def main():
    choice = input("Choose detection mode (1 for Image, 2 for Camera): ")  # Ask user for detection mode

    if choice == '1':  # If user selects image mode
        image_path = input("Enter the path to the UNO card image: ").strip()  # Ask user for image file path
        detect_card_from_image(image_path)  # Call the function to detect card from image
    elif choice == '2':  # If user selects camera mode
        print("Starting camera detection... Press 'q' to quit.")  # Inform user about camera mode
        detect_card_from_camera()  # Call the function to detect card from camera feed
    else:
        print("Invalid choice. Please enter 1 or 2.")  # Handle invalid choice input

# Run the main function
if __name__ == "__main__":  # Ensure the script runs only when executed directly
    main()
