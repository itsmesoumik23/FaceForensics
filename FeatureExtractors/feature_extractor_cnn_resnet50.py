import os
import pandas as pd
from keras.applications import ResNet50
import numpy as np
import cv2
from sklearn.decomposition import PCA
from tensorflow.keras.applications.resnet import preprocess_input
# Load ResNet50 model with pre-trained weights
model = ResNet50(weights='imagenet', include_top=False)

# Define the folder paths
frames_folder = './frames'
subfolders = ['NeuralTextures', 'FaceSwap', 'Face2Face', 'Deepfakes']

# Define the number of components for PCA
n_components = 1500

# Define the columns for the CSV file
columns = ['category'] + [f'feature_{i}' for i in range(n_components)]

# Initialize a list to store the features and categories
features = []
categories = []

# Loop through the subfolders
for subfolder in subfolders:
    folder_path = os.path.join(frames_folder, subfolder)

    # Loop through the images in the subfolder
    for image_name in os.listdir(folder_path):
        # Read the image using cv2
        image_path = os.path.join(folder_path, image_name)
        image = cv2.imread(image_path)

        # Preprocess the image for ResNet50
        image = cv2.resize(image, (224, 224))
        image = preprocess_input(image)
        image = np.expand_dims(image, axis=0)

        # Extract features from the image using ResNet50
        feature = model.predict(image)
        feature = feature.flatten()
        features.append(feature)
        categories.append(subfolder)

df=pd.DataFrame(features)

# Fit the PCA model to the features
pca = PCA(n_components=n_components)
pca.fit(df)

# Transform the features to the specified number of components
components = pca.transform(df)

# Store the transformed features in a new data frame
df_pca = pd.DataFrame(components, columns=columns[1:])
df_pca['category'] = categories
df_pca = df_pca[['category'] + columns[1:]]
df_pca.to_csv('features_resnet_50.csv', index=False)
