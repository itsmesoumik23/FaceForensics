import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2

from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
import os
import seaborn as sns
from keras.applications.vgg16 import VGG16

# Read input images and assign labels based on folder names
print(os.listdir("images/"))

SIZE = 256  # Resize images

# Capture training data and labels into respective lists
train_images = []
train_labels = []

for directory_path in glob.glob("images/train/*"):
    label = directory_path.split("\\")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_images.append(img)
        train_labels.append(label)

# Convert lists to arrays
train_images = np.array(train_images)
train_labels = np.array(train_labels)

# Capture test/validation data and labels into respective lists

test_images = []
test_labels = []
for directory_path in glob.glob("images/validation/*"):
    fruit_label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        test_images.append(img)
        test_labels.append(fruit_label)

# Convert lists to arrays
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Encode labels from text to integers.
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)

# Split data into test and train datasets (already split but assigning to meaningful convention)
x_train, y_train, x_test, y_test = train_images, train_labels_encoded, test_images, test_labels_encoded

###################################################################
# Normalize pixel values to between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# One hot encode y values for neural network.
from keras.utils import to_categorical

y_train_one_hot = to_categorical(y_train)

#############################
# Load model wothout classifier/fully connected layers
VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))

# Make loaded layers as non-trainable. This is important as we want to work with pre-trained weights
for layer in VGG_model.layers:
    layer.trainable = False

VGG_model.summary()  # Trainable parameters will be 0

# Now, let us use features from convolutional network for RF
feature_extractor = VGG_model.predict(x_train)

features = feature_extractor.reshape(feature_extractor.shape[0], -1)

X_for_RF = features
print(X_for_RF)
print(len(X_for_RF))
for i in y_train:
    print(i)
print(len(y_train))
import csv

d = {0: "Deepfakes", 1: "Face2face", 2: "Faceswap", 3: "NeuralTextures"}

csvData = []
for line, i in zip(X_for_RF.tolist(), y_train.tolist()):
    newImg = line

    # Prepend the category of each image to
    # the beginning of the features
    newImg.insert(0, d[i])
    csvData.append(newImg)

# Save the csv file
with open('Features.csv', 'w', newline='') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(csvData)

csvFile.close()

print("Done Extracting Features")
