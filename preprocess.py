import time

from misc import read_img
from tqdm import tqdm
import cv2
import numpy as np

from skimage.filters import gaussian
from skimage import color, segmentation
from skimage.segmentation import slic

def remove_letters(img):
    hh, ww = img.shape[:2]

    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # apply otsu thresholding
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]

    # apply morphology close to remove small regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # apply morphology open to separate breast from other regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    # get largest contour
    contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    big_contour = max(contours, key=cv2.contourArea)

    # draw largest contour as white filled on black background as mask
    mask = np.zeros((hh, ww), dtype=np.uint8)
    cv2.drawContours(mask, [big_contour], 0, 255, cv2.FILLED)

    # dilate mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (55, 55))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    # apply mask to image
    result = cv2.bitwise_and(img, img, mask=mask)
    return result

def remove_lines(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # lower mask (0-10)
    lower_red = np.array([0, 0, 0])
    upper_red = np.array([10, 255, 255])
    mask0 = cv2.inRange(hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([160, 0, 0])
    upper_red = np.array([179, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    # join my masks
    mask = mask0 + mask1

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilate = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)

    result = gray.copy()

    # Substitute colour for dilatation in colour position

    temp = dilate[mask == 0]
    result[mask == 0] = temp

    # Get rid of central line

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilate = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

    # Subtract dilatation for rest of image, get center ring as white
    center_line_mask = cv2.subtract(dilate, result)

    # Eliminate a bit of the subtraction noise, not belonging to center line
    ret, thresh_img = cv2.threshold(center_line_mask, 28, 255, cv2.THRESH_TOZERO)

    # Add center line
    result = cv2.addWeighted(result, 1, thresh_img, 1, 0.0)

    # result3 = cv2.cvtColor(result3, cv2.COLOR_GRAY2RGB)
    return result

def preprocess(img_size):

    # Globals declaration
    global path_benign_img
    path_benign_img = './resources/Mini_DDSM_Upload/Benign/'
    global path_cancer_img
    path_cancer_img = './resources/Mini_DDSM_Upload/Cancer/'
    global path_normal_img
    path_normal_img = './resources/Mini_DDSM_Upload/Normal/'

    img_list_benign = read_img(path_benign_img, 'benign')
    print("Benign images loaded")
    img_list_cancer = read_img(path_cancer_img, 'cancer')
    print("Cancer images loaded")
    img_list_normal = read_img(path_normal_img, 'normal')
    print("Normal images loaded")
    time.sleep(0.5)

    img_list_complete = dict(img_list_normal, **img_list_cancer)
    img_list_complete = dict(img_list_complete, ** img_list_benign)

    img_array_complete = []

    for file, vals in tqdm(img_list_complete.items()):

        img, tag = vals[0], vals[1]

        img = remove_letters(img)

        img = remove_lines(img)

        img = cv2.resize(img,img_size)
        img = img.reshape((img_size[0],img_size[1],1))

        #img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        cv2.imwrite('./resources/preprocess/' + tag + '/' + file, img)
        img_array_complete.append(img)

    return np.asarray(img_array_complete)