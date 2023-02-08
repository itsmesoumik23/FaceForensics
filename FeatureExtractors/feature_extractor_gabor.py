import csv
import cv2
import sys
import numpy as np
import os
from os import path
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import cmp_to_key

def cmp(a,b):
    return int(a[3:-4])-int(b[3:-4])


# Declare empty container to hold extracted category
category = []

# Declare empty container to hold image names
imagenames = []

# Declare empty container to hold extracted features
gaborArray = []

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
            img = cv2.resize(gray_img, (64, 128))

            df = pd.DataFrame()
            #Generate Gabor features
            num = 1  #To count numbers up in order to give Gabor features a lable in the data frame
            kernels = []
            for theta in range(2):   #Define number of thetas
                theta = theta / 4. * np.pi
                for sigma in (1, 3):  #Sigma with 1 and 3
                    for lamda in np.arange(0, np.pi, np.pi / 4):   #Range of wavelengths
                        for gamma in (0.05, 0.5):   #Gamma values of 0.05 and 0.5
                        
                            
                            gabor_label = 'Gabor' + str(num)  #Label Gabor columns as Gabor1, Gabor2, etc.
                            # print(gabor_label)
                            ksize=9
                            kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
                            kernels.append(kernel)
                            #Now filter the image and add values to a new column 
                            fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                            filtered_img = fimg.reshape(-1)
                            df[gabor_label] = filtered_img  #Labels columns as Gabor1, Gabor2, etc.
                            #print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                            num += 1  #Increment for gabor column label  

            #print(df)
            np_array=df.to_numpy()
            reshape_array=np_array.reshape(-1)

            # append the category of the
            # image to a category container
            category.append(folder)

            imagenames.append(filename)

            # append the extracted features of
            # the image to a category container
            gaborArray.append(reshape_array)


# convert the extracted features
# from array to vector
gaborArray_np = np.array(gaborArray)



# Create a container to hold data to be saved into csv
csvData = []
for id, line in enumerate(gaborArray_np.tolist()):
    newImg = line

    # Prepend the category of each image to
    # the begining of the features
    newImg.insert(0, category[id])
    newImg.insert(1, imagenames[id])
    csvData.append(newImg)


# Save the csv file
with open('Features_gabor.csv', 'w' ,newline='') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(csvData)

csvFile.close()

print("Done Extracting Features")

