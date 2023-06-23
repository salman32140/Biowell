import os
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.applications.inception_v3 import InceptionV3
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime


# Load the CSV file containing the ground truth
csv_file = './data/annotations.csv'
df = pd.read_csv(csv_file)

# Create a dictionary to map class names to class IDs
class_dict = {
    'Low Stress': 0,
    'Moderate Stress': 1,
    # Add more class names as needed
}


# Set up data loading and preprocessing

image_folder = './data/images/'
image_files = os.listdir(image_folder)
image_paths = [os.path.join(image_folder, file) for file in image_files]

# Split the dataset into train, test, and validation sets
train_paths, test_paths, train_labels, test_labels = train_test_split(
    image_paths, df['class_name'], test_size=0.2, random_state=42)

train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_paths, train_labels, test_size=0.2, random_state=42)

# Define data loading functions
def load_and_preprocess_image(path):
    image = load_img(path, target_size=(299, 299))
    image_array = img_to_array(image)
    return preprocess_input(image_array)

def load_data(paths, labels):
    images = [load_and_preprocess_image(path) for path in paths]
    labels = [class_dict[label] for label in labels]
    labels = to_categorical(labels, num_classes=len(class_dict))
    return tf.data.Dataset.from_tensor_slices((images, labels))

# Create TensorFlow data loaders
batch_size = 32

train_data = load_data(train_paths, train_labels)
train_data = train_data.shuffle(len(train_paths)).batch(batch_size)

val_data = load_data(val_paths, val_labels)
val_data = val_data.batch(batch_size)

test_data = load_data(test_paths, test_labels)
test_data = test_data.batch(batch_size)


num_classes = len(class_dict)
model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
output = tf.keras.layers.Dense(num_classes, activation='softmax')(model.output)
model = tf.keras.Model(model.input, output)

# Create TensorBoard callback
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


# Train and evaluate the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_data, validation_data=val_data, epochs=200, callbacks=[tensorboard_callback])

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_data)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_accuracy)

# Make predictions on the test set
y_true = np.concatenate([y for x, y in test_data], axis=0)
y_pred_prob = model.predict(test_data)
y_pred = np.argmax(y_pred_prob, axis=1)

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Create classification report
class_names = list(class_dict.keys())
report = classification_report(y_true, y_pred, target_names=class_names)

# Print confusion matrix and classification report
print('Confusion Matrix:')
print(cm)
print('---')
print('Classification Report:')
print(report)
    