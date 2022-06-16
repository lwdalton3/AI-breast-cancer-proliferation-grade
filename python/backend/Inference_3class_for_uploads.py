#Developed by Dalton Pathology_year_2020
# Any use should cite
# https://keras.io/api/applications/
# to run from console  python3 Inference_.py

#!/usr/bin/env python
# coding: utf-8

# In[6]:
## if no work may have to sudo pkill -9  in terminal

from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import os
import cv2
from tqdm import tqdm
import shutil
from statistics import mean, median, median_high


# In[8]:
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
try: tf.config.experimental.set_memory_growth(physical_devices[0], True)
except: pass # Invalid device or cannot modify virtual devices once initialized.

#===Inputs===#images

modelPath = "ResNet101V2_TCGA_orig_train.h5"   # model name change names as needed

biasToHighClass = 0 #Enter a value between 0 and 1. 0 means, no bias and 1 means, 100% of the time high class is predicted. You can try a value in between, if you want like 0.1,0.124,0.23 etc.

imageSize = 224 # Set Images size corresponding to the model

testFolder= "images" # Folder contains images that you would like to test

outputDir = "test_results"   # will place copy of image in folder according to classification

classNames = ["high","low","stroma"]             # Write the three class that were used in alphabetical order

#===Inputs===#

model = load_model(modelPath)

print("Loading file paths")
allFiles = []
for dirname, _, filenames in os.walk(testFolder): # We will us OS.Walk Function to get all file names inside this folders
    for filename in filenames:
        allFiles.append(os.path.join(dirname, filename))

print("Loaded file names")
def processIMG(i):
    global df
    global allFiles

    imagePath = allFiles[i]
    img2 = cv2.imread(imagePath)
    y, x, z = img2.shape
    current_x=0
    current_y=0
    list_img = [] # List of tiles
    start_point = [] # List of starting points of tiles
    end_point = [] # List of ending points of tiles

    #===Cropping images into (250, 250) tiles===#
    while current_x+250 <=x and current_y+250 <=y  :
        img_crop = img2[current_y:current_y+250,current_x:current_x+250]
        start_point.append((current_y, current_x))
        end_point.append((current_y+250, current_x+250))
        current_x += 250
        if current_x+250>x:
            current_x=0
            current_y+=250
        list_img.append(img_crop)
    #===Cropping images into (250, 250) tiles===#

    if img2 is None: #Image is None
        return 0
    high = 0
    low = 0
    stroma = 0
    for a in range(len(list_img)):
        #===Preprocessing of an images===#
        img_ori = list_img[a]
        img = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (imageSize,imageSize), interpolation = cv2.INTER_AREA) #Resize Image
        img = img/255 #Normalize image
        img = img.reshape(1,img.shape[0],img.shape[1],3)
        #===Preprocessing of an images===#

        predict = model.predict(img)
        predict[0][0] -= biasToHighClass #Add high class bias

        #===Get Predictions of tiles==#
        if predict[0][0] >0.5: # Return predictions of High
            high += 1
            cv2.rectangle(img2, start_point[a], end_point[a], (0,0,0), 2) ### if high draw black rectangle
        elif predict[0][1] >0.5: # Return predictions of Low
            low += 1
        elif predict[0][2] >0.5: # Return predictions of Stroma
            stroma += 1
        #===Get Predictions of tiles==#

    #===Print out results for each images===#
    print(filename+": ")
    print("high: " +str(high))
    print("low: " +str(low))
    print("stroma: " +str(stroma))
    #===Print out results for each images===#

    #===Show the results images with rectangles for each images===#
    #  cv2.imshow(imagePath, img2)
    cv2.imwrite(imagePath+"_result.png", img2)
    #  cv2.waitKey()
    #===Show the results images with rectangles for each images===#

for i in tqdm(range(len(allFiles))):
    processIMG(i)
