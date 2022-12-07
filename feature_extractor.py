import csv
import cv2
import sys
import numpy as np
import os
from os import path
import sys
import numpy as np
from tqdm import tqdm
from functools import cmp_to_key

hog = cv2.HOGDescriptor()

# Declare empty container to hold extracted category
category = []

# Declare empty container to hold image names
imageNames = []

# Declare empty container to hold extracted features
hogArray = []


for folder in tqdm(os.listdir("./frames/")):

    # Loop through each category
    arr = os.listdir(path.join("./frames/", folder))
    # print(arr)
    arr.sort(key=lambda x: int(x[3:-4]))
    # print(arr)
    for filename in arr:

        # Select images which are png and jpg only
        if filename[-3:] == "png" or filename[-3:] == "jpg":

            # Get full image by joining
            # all the path to the image
            image = path.join("./frames", folder, filename)

            # Use open cv to read the image
            img = cv2.imread(image)

            # Resize the image to (64, 128)
            # Default for hog
            resized = cv2.resize(img, (64, 128), interpolation=cv2.INTER_AREA)

            # Compute the Hog Features
            h = hog.compute(resized)

            # Transpose the result from a vector to an array
            hogImage = h.T

            # append the category of the
            # image to a category container
            category.append(folder)

            imageNames.append(filename)

            # append the extracted features of
            # the image to a category container
            hogArray.append(hogImage)


# convert the extracted features
# from array to vector
hogArray_np = np.array(hogArray)
features = hogArray_np.tolist()

# Create a container to hold data to be saved into csv
csvData = []
for id, line in enumerate(features):
    newImg = line

    # Prepend the category of each image to
    # the beginning of the features
    newImg.insert(0, category[id])
    newImg.insert(1, imageNames[id])
    csvData.append(newImg)


# Save the csv file
with open('Features.csv', 'w', newline='') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(csvData)

csvFile.close()

print("Done Extracting Features")
