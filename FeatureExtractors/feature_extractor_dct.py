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

def cmp(a,b):
    return int(a[3:-4])-int(b[3:-4])


# Declare empty container to hold extracted category
category = []

# Declare empty container to hold image names
imagenames = []

# Declare empty container to hold extracted features
dctArray = []

for folder in tqdm(os.listdir("./frames/")):

    # Loop through each category
    arr=os.listdir(path.join("./frames/", folder))
    arr.sort(key=cmp_to_key(cmp))
    for filename in arr:

        # Select images which are png and jpg only
        if (filename[-3:] == "png" or filename[-3:] == "jpg"):

            # Get full image by joining
            # all the path to the image
            image = path.join("./frames", folder, filename)

            # Use open cv to read the image
            img = cv2.imread(image)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Resize the image to (64, 128)
            resized = cv2.resize(gray_img, (64, 128))

            dctimg= cv2.dct(np.float32(resized))

            feature_dct=[]
            for row in dctimg:
            	for col in row:
            		feature_dct.append(col)


            # append the category of the
            # image to a category container
            category.append(folder)

            imagenames.append(filename)

            # append the extracted features of
            # the image to a category container
            dctArray.append(feature_dct)


# convert the extracted features
# from array to vector
dctArray_np = np.array(dctArray)



# Create a container to hold data to be saved into csv
csvData = []
for id, line in enumerate(dctArray_np.tolist()):
    newImg = line

    # Prepend the category of each image to
    # the begining of the features
    newImg.insert(0, category[id])
    newImg.insert(1, imagenames[id])
    csvData.append(newImg)


# Save the csv file
with open('Features_dct.csv', 'w' ,newline='') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(csvData)

csvFile.close()

print("Done Extracting Features")





# # Load the input image
# inp_img = cv2.imread("./frames/Deepfakes/df_0.jpg")

# # Convert the image to grayscale
# gray_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2GRAY)


# # Apply cv2.dct to the grayscale image
# dct = cv2.dct(np.float32(gray_img))

# # Extract the features into an array
# features = np.array(dct)

# # Output the array of features
# print(features)