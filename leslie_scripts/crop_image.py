"""
Performs image segmentation.
"""

import cv2
import os
from tqdm import tqdm


input_dir = 'test'
output_dir = 'test_panels'
all_files = os.listdir(input_dir)


def crop_image(i):
    img = cv2.imread(f'./{input_dir}/{all_files[i]}')
    y, x, z = img.shape
    current_x = 0
    current_y = 0
    list_img = []
    while current_x + 250 <= x and current_y + 250 <= y:
        img_crop = img[current_y:current_y + 250, current_x:current_x + 250]
        current_x += 250
        if current_x + 250 > x:
            current_x = 0
            current_y += 250
        list_img.append(img_crop)

    for i in range(len(list_img)):
        cv2.imwrite(f'./{output_dir}/img_crop{i}.jpeg', list_img[i])


for i in tqdm(range(len(all_files))):
    crop_image(i)
