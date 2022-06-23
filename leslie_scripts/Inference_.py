#Developed by Dalton Pathology_year_2020
# Any use should cite
#
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


# In[8]:
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
try: tf.config.experimental.set_memory_growth(physical_devices[0], True)
except: pass # Invalid device or cannot modify virtual devices once initialized.

#===Inputs===#images
#modelPath= "VGG19_CENPA_125_train.h5"ResNet101V2_model_TCGA_big_may_7_train
modelPath = "VGG19_CHTN_train_images_may_day.h5"   # model name change names as needed
#modelPath = "/home/ldalton/ResNet101V2_TCGA_orig_train.h5"   # model name change names as needed

biasToHighClass = 0 #Enter a value between 0 and 1. 0 means, no bias and 1 means, 100% of the time high class is predicted. You can try a value in between, if you want like 0.1,0.124,0.23 etc.
imageSize = 224

#testFolder = "TCGA_screenshots_125"                            # change folder as needed
#testFolder= "/home/ldalton/TCGA_screenshots_pleo1_125"

testFolder= "test_panels"
#outputDir = "/home/ldalton/results_TCGA_screenshots_pleo1_125"   # will place copy of image in folder according to classification

outputDir = "test_results"   # will place copy of image in folder according to classification
#outputDir = "VGG19_CENPA_125_train_CHTN_test_results"        Hamburg_scshots_e_results_TCGA_orig          # change name as needed to reflect what is being done
csvResultPath = "test_results.tsv"          # text file with data on each image
#csvResultPath = "/home/ldalton/can_delete.tsv"          # text file with data on each image


classNames = ["high","low","stroma"]                            # three classes needed
#===Inputs===#


model = load_model(modelPath)

#Create empty CSV file to store results.
df = pd.DataFrame(columns=['ID','confidence','prediction','high_confidence','low_confidence','stroma_confidence'])


print("Loading file paths")
allFiles = []
for dirname, _, filenames in os.walk(testFolder): #WRITE TEST FOLDER NAME HERE #We will us OS.Walk Function to get all file names inside this 4 folders
    for filename in filenames:
        allFiles.append(os.path.join(dirname, filename))


print("Loaded file names")
def processIMG(i):
    global df
    global allFiles

    imagePath = allFiles[i]
    img = cv2.imread(imagePath)
    if img is None: #Image is None
        return 0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (imageSize,imageSize), interpolation = cv2.INTER_AREA) #Resize Image
    img = img/255 #Normalize image
    img = img.reshape(1,img.shape[0],img.shape[1],3)
    predict = model.predict(img)
    predict[:,0] += biasToHighClass #Add high class bias
    probabilities = predict[0].copy()
    maxProbability = int(max(predict[0])*100)
    predict = np.argmax(predict) #Get predicted class ID/Index

    name = classNames[predict] #Get predicted class name

    saveDir = os.path.join(outputDir,name)
    if os.path.isdir(saveDir) == False: #If directory do not exist, make one
        os.makedirs(saveDir)
        print("Creating:",saveDir)

    #Append result to CSV file
    df = df.append(pd.DataFrame([[os.path.basename(allFiles[i]),maxProbability,name,probabilities[0],probabilities[1],probabilities[2]]], columns=df.columns))
    #Copy image to class folder.
    shutil.copy(allFiles[i],saveDir)

for i in tqdm(range(len(allFiles))):
    processIMG(i)


df.to_csv(csvResultPath,index=False, sep="\t")


# In[ ]:





# In[ ]:




