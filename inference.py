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


def predict(input_path, model, bias_to_high_class=0, image_size=224,
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

    # Check whether we're processing a directory of images or a single image
    if os.path.isdir(input_path):
        folder_content = os.listdir(input_path)
        image_names = [os.path.join(input_path, image_name) for image_name in
                       folder_content]
    else:
        image_names = [input_path]

    # Output TSV
    df = pd.DataFrame(columns=['ID', 'confidence', 'prediction',
                               'high_confidence', 'low_confidence',
                               'stroma_confidence'])

    # Process images
    for image_name in image_names:
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

            prediction = model.predict(img)

            # Add high class bias
            prediction[0][0] -= bias_to_high_class

            rect_for_blending = np.ones(img_ori.shape, dtype=np.uint8)

            # Get predictions of tiles
            # Return predictions of High
            label = labels[prediction.argmax()]
            confidence = prediction.max()

            # Map confidence to exp scale so that the coloring differene is
            # more drastic
            confidence = (np.exp(confidence) - 1)/(np.e - 1)

            # Write image to disk
            cv2.imwrite(os.path.join(output_path, label.lower(), segment_name),
                        img_ori)

            if label == 'HIGH':
                high += 1
                rect_for_blending = (rect_for_blending*high_highlight*confidence
                                     ).astype(np.uint8)
                res = cv2.addWeighted(img_ori, 0.7, rect_for_blending, 0.5,
                                      1.0)
                img_ori[:] = res

            # Return predictions of Low
            elif label == 'LOW':
                low += 1
                rect_for_blending = (rect_for_blending*low_stroma_highlight*confidence
                                     ).astype(np.uint8)
                res = cv2.addWeighted(img_ori, 0.7, rect_for_blending, 0.5,
                                      1.0)
                img_ori[:] = res

            # Return predictions of Stroma
            elif label == 'STROMA':
                stroma += 1
                rect_for_blending = (rect_for_blending*low_stroma_highlight*confidence
                                     ).astype(np.uint8)
                res = cv2.addWeighted(img_ori, 0.7, rect_for_blending, 0.5,
                                      1.0)
                img_ori[:] = res

            # Append prediction data to TSV
            probabilities = prediction[0].copy()
            max_probability = int(max(prediction[0])*100)
            df = df.append(pd.DataFrame([[segment_name, max_probability, label,
                                          probabilities[0], probabilities[1],
                                          probabilities[2]]],
                                        columns=df.columns))

        # Save the image
        cv2.imwrite(os.path.join(output_path, f'{save_name}_result.png'), img2)

    # Save the TSV
    if os.path.isdir(input_path):
        df.to_csv(os.path.join(os.path.dirname(output_path), 'result.tsv'),
                  index=False, sep="\t")
    else:
        df.to_csv(os.path.join(output_path, f'{save_name}_result.tsv'),
                  index=False, sep="\t")


if __name__ == '__main__':

    # Load model
    model_path = 'VGG19_CHTN_train_images_may_day.h5'
    model = load_model(model_path)

    # 1. Analyse single image
    #  predict('/home/ldalton/test/ska.jpg', model, output_dir='./output')

    # 2. Analyse directory of images, and save TSV results separately for each
    # image
    #  dir_name = '/home/ldalton/test'
    #  images = [os.path.join(dir_name, image_name) for image_name in
              #  os.listdir(dir_name)]
    #  for image_name in images:
        #  predict(image_name, model, output_dir='./output')

    # 3. Analyse directory of images, and save results of all images in a
    # single TSV
    predict('/home/ldalton/test', model, output_dir='./output')
