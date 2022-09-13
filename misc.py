import os
import random

import cv2

def read_img(path, tag):
    img_list = {}
    files = os.listdir(path)
    max_sample_size = 100
    i = 0
    for file in files:
        if i > max_sample_size:
            break
        if os.path.isfile(os.path.join(path, file)):
            img = cv2.imread(os.path.join(path, file))
            img_list.update({file: [img, tag]})
        i += 1
    return img_list