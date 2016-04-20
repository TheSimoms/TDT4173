import os
import numpy

import Image

from sklearn.preprocessing import scale

from skimage.filter import scharr
from skimage.transform import rescale


def load_image(directory_name, file_name):
    letter = directory_name.split('/')[-1].lower()
    correct_label = ord(letter) - 97

    img = Image.open('%s/%s' % (directory_name, file_name))

    return numpy.array(img, dtype="float64"), correct_label


def load_data():
    data = []
    labels = []

    for root, directory_names, files in os.walk('data', topdown=False):
        for directory_name in directory_names:
            path = '%s/%s' % (root, directory_name)

            for file_name in os.listdir(path):
                img = load_image(path, file_name)

                data.append(img[0])
                labels.append(img[1])

    return data, labels


def pre_process_data(feature_sets):
    for i in range(len(feature_sets)):
        feature_sets[i] = scale(feature_sets[i])
        feature_sets[i] = scharr(feature_sets[i])

    return feature_sets


def sliding_windows(feature_set):
    windows = []
    window_information = []

    for image_scale in numpy.arange(1, 3):
        temp_image = rescale(feature_set, image_scale)

        height, width = temp_image.shape

        for y in range(0, height-20+1, 10):
            for x in range(0, width-20+1, 10):
                windows.append(temp_image[x:x+20, y:y+20].flatten())
                window_information.append(((x, y), image_scale))

    return windows, window_information
