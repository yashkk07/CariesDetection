import os
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Define dataset paths
dataset_path="teeth_dataset"
training_path="teeth_dataset/Trianing"
testing_path="teeth_dataset/test"
caries_path="teeth_dataset/Trianing/caries"
without_caries_path="teeth_dataset/Trianing/without_caries"
test_caries_path="teeth_dataset/test/caries"
test_without_caries_path="teeth_dataset/test/no-caries"
# Class labels
class_labels = {"caries": 0, "no_caries": 1}

# Image dimensions for resizing
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalize pixel values to [0, 1]
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Split training data into train/validation sets
)

# Prepare training and validation datasets
train_generator = train_datagen.flow_from_directory(
    training_path,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    training_path,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# Data generator for test dataset
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory(
    testing_path,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Build the CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    verbose=1
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save the model
model.save("teeth_caries_classifier.keras")
print("Model saved as 'teeth_caries_classifier.keras'.")

# Plot training and validation accuracy/loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.legend()

plt.show()

# Function to display sample images
def display_sample_images(folder_path, title, num_samples=5):
    plt.figure(figsize=(15, 5))
    image_files = os.listdir(folder_path)

    for i in range(num_samples):
        img_path = os.path.join(folder_path, random.choice(image_files))
        img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = img_to_array(img) / 255.0  # Normalize the image

        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img_array)
        plt.axis('off')
        plt.title(title)

# Display samples from the "caries" class
display_sample_images(caries_path, "Caries", num_samples=5)

# Display samples from the "no-caries" class
display_sample_images(without_caries_path, "No Caries", num_samples=5)

plt.tight_layout()
plt.show()

# Load the trained model
model = tf.keras.models.load_model("teeth_caries_classifier.keras")
print("Model loaded successfully.")

# Function to preprocess and predict an image
def predict_image_class(img_path, model, threshold=0.5):
    img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)
    class_idx = int(prediction[0][0] > threshold)  # Threshold: 0.5
    return class_idx, prediction[0][0]  # Returns class index and probability

# Function to generate predictions for a folder
def generate_predictions(folder_path, model, class_labels, threshold=0.5):
    predictions = []
    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        class_idx, prob = predict_image_class(img_path, model, threshold)
        predicted_class = [k for k, v in class_labels.items() if v == class_idx][0]
        predictions.append((img_file, predicted_class, prob))
    return predictions

# Generate predictions for "caries" images in the test set
caries_predictions = generate_predictions(test_caries_path, model, class_labels)
print("\nPredictions for Caries Images:")
for img_name, pred_class, prob in caries_predictions:
    print(f"Image: {img_name}, Predicted Class: {pred_class}, Probability: {prob:.2f}")

# Generate predictions for "no-caries" images in the test set
no_caries_predictions = generate_predictions(test_without_caries_path, model, class_labels)
print("\nPredictions for No-Caries Images:")
for img_name, pred_class, prob in no_caries_predictions:
    print(f"Image: {img_name}, Predicted Class: {pred_class}, Probability: {prob:.2f}")


# Count the images in caries and without caries directories
caries_count = len(os.listdir(caries_path))
without_caries_count = len(os.listdir(without_caries_path))

# Data for the bar graph
categories = ['Caries', 'Without Caries']
counts = [caries_count, without_caries_count]

# Plotting the bar graph
plt.figure(figsize=(8, 6))
plt.bar(categories, counts, color=['red', 'green'])
plt.title('Count of Images in Training Dataset', fontsize=16)
plt.xlabel('Category', fontsize=14)
plt.ylabel('Number of Images', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adding count labels above bars
for i, count in enumerate(counts):
    plt.text(i, count + 2, str(count), ha='center', fontsize=12, color='black')

plt.show()

# Generate prediction probabilities for the test dataset
def get_prediction_probabilities(generator, model):
    probabilities = []
    for batch, _ in generator:
        batch_probabilities = model.predict(batch)
        probabilities.extend(batch_probabilities.flatten())  # Flatten to 1D list
        if len(probabilities) >= generator.samples:  # Stop after processing all samples
            break
    return probabilities

# Get prediction probabilities from the test generator
test_probabilities = get_prediction_probabilities(test_generator, model)

# Plot the histogram of prediction probabilities
plt.figure(figsize=(8, 6))
plt.hist(test_probabilities, bins=20, color='blue', alpha=0.7, edgecolor='black')
plt.title('Distribution of Prediction Probabilities', fontsize=16)
plt.xlabel('Prediction Probability', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the plot
histogram_path = os.path.join(dataset_path, 'plots/probability_histogram.png')
plt.savefig(histogram_path)
plt.show()

print(f"Prediction probability histogram saved to {histogram_path}")

# Get true labels and predictions for the test dataset
true_labels = test_generator.classes
predictions = model.predict(test_generator)
predicted_labels = (predictions > 0.5).astype(int).flatten()  # Thresholding at 0.5

# Calculate the confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels.keys())

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
disp.plot(cmap='Blues', values_format='d', ax=plt.gca())
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Predicted Labels', fontsize=14)
plt.ylabel('True Labels', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Save the plot
confusion_matrix_path = os.path.join(dataset_path, 'plots/confusion_matrix.png')
plt.savefig(confusion_matrix_path)
plt.show()

print(f"Confusion matrix plot saved to {confusion_matrix_path}")
