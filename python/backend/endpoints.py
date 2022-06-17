from threading import Thread
import cv2
import werkzeug
import tensorflow as tf
from flask import send_file
from tensorflow.keras.models import load_model
from flask_restful import Resource, reqparse


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

# Will place copy of image in folder according to classification
outputDir = "test_results"

# Write the three class that were used in alphabetical order
classNames = ["high", "low", "stroma"]

model = load_model(modelPath)

threads = []

high = 0
low = 0
stroma = 0


def predict(image):
    """
    Predicts on a single image.
    """

    img2 = cv2.imread(image)
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

    if img2 is None:
        return 0

    global high
    global low
    global stroma

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

        # Get predictions of tiles
        # Return predictions of High
        if predict[0][0] > 0.5:
            high += 1

            # If high draw black rectangle
            cv2.rectangle(img2, start_point[a], end_point[a], (0, 0, 0), 2)

        # Return predictions of Low
        elif predict[0][1] > 0.5:
            low += 1

        # Return predictions of Stroma
        elif predict[0][2] > 0.5:
            stroma += 1

    cv2.imwrite('result.png', img2)


class Uploader(Resource):
    def get(self):
        return {'hello': 'world'}

    def post(self):
        parse = reqparse.RequestParser()
        parse.add_argument('filepond',
                           type=werkzeug.datastructures.FileStorage,
                           location='files')
        args = parse.parse_args()
        image_file = args['filepond']
        image_file.save('temp_image.jpg')

        threads.append(Thread(target=predict, args=('temp_image.jpg',)))
        threads[-1].start()

        # Return result of prediction
        return {'status': 'Prediction started'}


class PredictionStatus(Resource):
    def get(self):
        # Check if thread has finished
        if threads[-1].is_alive():
            return {'status': 'Thread still running.'}
        else:
            return {'status': 'Prediction complete.',
                    'data': {'high': high, 'low': low, 'stroma': stroma}}


class DownloadImage(Resource):
    def get(self):
        return send_file('result.png', mimetype='image/gif')
