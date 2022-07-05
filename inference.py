"""
Performs inference on images of cells, trying to detect the presence of breast
cancer.

Works on one image at a time. If you don't specify a custom output folder, the
output will be stored in the directory of the image.
"""

import os
import shutil
import cv2
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


physical_devices = tf.config.list_physical_devices('GPU')

try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass


def predict(image_name, model, bias_to_high_class=0, image_size=224,
            output_dir=None):
    """
    Predicts on a single image with name `image_name` and stores the results in
    a directory called `image_name_inference_result`.

    Parameters
    ----------
        image_name : str
            Path to the image to perform inference on.

        model : tf.keras.Model instance
            Keras model to use for inference.

        bias_to_high_class : float
            A value between 0 and 1, where 0 means, no bias and 1 means, 100%
            of the time high class is predicted.

        image_size : int
            Resolution to which the input image will be resized
            (`image_size`x`image_size`).

        output_dir : str
            Directory in which to store inference results. If it's not
            specified, the results will be stored in the directory of the input
            image.
    """

    img2 = cv2.imread(image_name)
    y, x, z = img2.shape
    current_x = 0
    current_y = 0
    labels = ['HIGH', 'LOW', 'STROMA']

    # List of tiles
    list_img = []

    # List of starting points of tiles
    start_point = []

    # List of ending points of tiles
    end_point = []

    # Used for counting image segments per class
    high = 0
    low = 0
    stroma = 0

    # Colors for highlighting tiles after classification
    scale = 0.7
    high_highlight = np.array([255, 0, 0])[::-1]*scale
    low_stroma_highlight = np.array([0, 0, 255])[::-1]*scale

    # Make directory structure for output
    save_name = os.path.splitext(os.path.basename(image_name))[0]
    image_path_no_prefix = os.path.splitext(image_name)[0]
    if output_dir:
        output_path = os.path.join(output_dir, f'{save_name}_results')
    else:
        output_path = os.path.join(f'{image_path_no_prefix}_results')

    if os.path.exists(output_path):
        if os.path.isdir(output_path):
            shutil.rmtree(output_path)
        else:
            os.remove(output_path)

    os.makedirs(output_path)

    # Add class folders
    os.makedirs(os.path.join(output_path, 'high'))
    os.makedirs(os.path.join(output_path, 'low'))
    os.makedirs(os.path.join(output_path, 'stroma'))

    # Output TSV
    df = pd.DataFrame(columns=['ID', 'confidence', 'prediction',
                               'high_confidence', 'low_confidence',
                               'stroma_confidence'])

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

    for a in range(len(list_img)):

        segment_name = f'{save_name}_segment_{a}.png'

        # Preprocessing
        img_ori = list_img[a]
        img = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)

        # Resize Image
        img = cv2.resize(img, (image_size, image_size),
                         interpolation=cv2.INTER_AREA)

        # Normalize image
        img = img/255
        img = img.reshape(1, img.shape[0], img.shape[1], 3)

        predict = model.predict(img)

        # Add high class bias
        predict[0][0] -= bias_to_high_class

        rect_for_blending = np.ones(img_ori.shape, dtype=np.uint8)

        # Get predictions of tiles
        # Return predictions of High
        label = labels[predict.argmax()]
        confidence = predict.max()

        # Map confidence to exp scale so that the coloring differene is more
        # drastic
        confidence = (np.exp(confidence) - 1)/(np.e - 1)

        # Write image to disk
        cv2.imwrite(os.path.join(output_path, label.lower(), segment_name),
                    img_ori)

        if label == 'HIGH':
            high += 1
            rect_for_blending = (rect_for_blending*high_highlight*confidence
                                 ).astype(np.uint8)
            res = cv2.addWeighted(img_ori, 0.7, rect_for_blending, 0.5, 1.0)
            img_ori[:] = res

        # Return predictions of Low
        elif label == 'LOW':
            low += 1
            rect_for_blending = (rect_for_blending*low_stroma_highlight*confidence
                                 ).astype(np.uint8)
            res = cv2.addWeighted(img_ori, 0.7, rect_for_blending, 0.5, 1.0)
            img_ori[:] = res

        # Return predictions of Stroma
        elif label == 'STROMA':
            stroma += 1
            rect_for_blending = (rect_for_blending*low_stroma_highlight*confidence
                                 ).astype(np.uint8)
            res = cv2.addWeighted(img_ori, 0.7, rect_for_blending, 0.5, 1.0)
            img_ori[:] = res

        # Append prediction data to TSV
        probabilities = predict[0].copy()
        max_probability = int(max(predict[0])*100)
        df = df.append(pd.DataFrame([[segment_name, max_probability, label,
                                      probabilities[0], probabilities[1],
                                      probabilities[2]]], columns=df.columns))

    # Show inference result
    #  cv2.imshow("Inference Result", img2)
    #  cv2.waitKey(0)

    # Save the image
    cv2.imwrite(os.path.join(output_path, f'{save_name}_result.png'), img2)

    # Save the TSV
    df.to_csv(os.path.join(output_path, f'{save_name}_result.tsv'),
              index=False, sep="\t")


if __name__ == '__main__':

    # Load model
    model_path = 'VGG19_CHTN_train_images_may_day.h5'
    model = load_model(model_path)

    # Analyse single image (with custom output dir)
    predict('/home/ldalton/test/ska.jpg', model, output_dir='./output')

    # Can also be used with directory of images
    #  dir_name = './sample_data'
    #  images = [os.path.join(dir_name, image_name) for image_name in
              #  os.listdir(dir_name)]

    #  for image_name in images:
        #  predict(image_name, model)
