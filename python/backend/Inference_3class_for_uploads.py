import os
import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tqdm import tqdm


physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

modelPath = "ResNet101V2_TCGA_orig_train.h5"

# Enter a value between 0 and 1. 0 means, no bias and 1 means, 100% of the time
# high class is predicted. You can try a value in between, if you want like
# 0.1,0.124,0.23 etc.
biasToHighClass = 0

# Set Images size corresponding to the model
imageSize = 224

# Folder contains images that you would like to test
testFolder = "images"

model = load_model(modelPath)

allFiles = []

# Read images from directory
for dirname, _, filenames in os.walk(testFolder):
    for filename in filenames:
        allFiles.append(os.path.join(dirname, filename))


def processIMG(imagePath):
    global df
    global allFiles

    img2 = cv2.imread(imagePath)
    y, x, z = img2.shape
    current_x = 0
    current_y = 0

    # List of tiles
    list_img = []

    # List of starting points of tiles
    start_point = []

    # List of ending points of tiles
    end_point = []

    # Cropping images into (250, 250) tiles
    while current_x + 250 <= x and current_y + 250 <= y:
        img_crop = img2[current_y:current_y + 250, current_x:current_x + 250]
        start_point.append((current_y, current_x))
        end_point.append((current_y + 250, current_x + 250))
        current_x += 250
        if current_x + 250 > x:
            current_x = 0
            current_y += 250
        list_img.append(img_crop)

    # Colors for highlighting tiles after classification
    #  low_highlight = np.array([230, 229, 241])[::-1]
    #  high_highlight = np.array([230, 214, 213])[::-1]
    #  stroma_highlight = np.array([229, 238, 226])[::-1]
    low_highlight = np.array([0, 255, 0])[::-1]
    high_highlight = np.array([255, 0, 0])[::-1]
    stroma_highlight = np.array([0, 0, 255])[::-1]

    high = 0
    low = 0
    stroma = 0

    for a in range(len(list_img)):
        # Preprocessing
        img_ori = list_img[a]
        img = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)

        # Resize Image
        img = cv2.resize(img, (imageSize, imageSize),
                         interpolation=cv2.INTER_AREA)

        # Normalize image
        img = img/255
        img = img.reshape(1, img.shape[0], img.shape[1], 3)

        predict = model.predict(img)

        # Add high class bias
        predict[0][0] -= biasToHighClass

        rect_for_blending = np.ones(img_ori.shape, dtype=np.uint8)

        # Get predictions of tiles
        # Return predictions of High
        if predict[0][0] > 0.5:
            high += 1

            rect_for_blending = (rect_for_blending*high_highlight).astype(
                np.uint8)
            res = cv2.addWeighted(img_ori, 0.5, rect_for_blending, 0.5, 1.0)
            img_ori[:] = res

        # Return predictions of Low
        elif predict[0][1] > 0.5:
            low += 1

            rect_for_blending = (rect_for_blending*low_highlight).astype(
                np.uint8)
            res = cv2.addWeighted(img_ori, 0.5, rect_for_blending, 0.5, 1.0)
            img_ori[:] = res

        # Return predictions of Stroma
        elif predict[0][2] > 0.5:
            stroma += 1

            rect_for_blending = (rect_for_blending*stroma_highlight).astype(
                np.uint8)
            res = cv2.addWeighted(img_ori, 0.5, rect_for_blending, 0.5, 1.0)
            img_ori[:] = res

    # Print out results for each images
    print("high: " + str(high))
    print("low: " + str(low))
    print("stroma: " + str(stroma))

    cv2.imwrite(imagePath + "_result.png", img2)
