import os
import numpy

import Image
import ImageDraw
import pickle

from sklearn.preprocessing import scale

from skimage.filter import scharr
from sklearn.cross_validation import train_test_split
from skimage.color import rgb2gray
from skimage.data import load


def flatten_feature_sets(feature_sets):
    return [feature_set.flatten() for feature_set in feature_sets]


def letter_to_index(letter):
    return ord(letter) - 97


def index_to_letter(index):
    return chr(index + 97)


def load_image(directory_name, file_name):
    letter = directory_name.split('/')[-1].lower()
    correct_label = letter_to_index(letter)

    return numpy.array(Image.open('%s/%s' % (directory_name, file_name)), dtype="float64"), correct_label


def load_arbitrary_image(image_path):
    return rgb2gray(load(image_path))


def extract_windows(image):
    windows = []
    window_positions = []

    height, width = image.shape

    window_width = 20
    window_height = 20

    for y in range(height-window_height+1):
        for x in range(width-window_width+1):
            windows.append(image[y:y+window_height, x:x+window_width])
            window_positions.append((x, y, window_width, window_height))

    return windows, window_positions


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


def draw_image_with_windows(image_path, windows):
    img = Image.open(image_path)

    draw = ImageDraw.Draw(img)

    for window in windows:
        x, y, window_width, window_height = window

        draw.line(
            (
                (x, y),
                (x+window_width, y),
                (x+window_width, y+window_height),
                (x, y+window_height),
                (x, y)
            ),
            fill=255
        )

    img.show()


def pre_process_data(feature_sets):
    for i in range(len(feature_sets)):
        feature_sets[i] = scale(feature_sets[i])
        feature_sets[i] = scharr(feature_sets[i])

    return feature_sets


def prepare_data():
    # Load data and labels
    data, labels = load_data()

    # Pre process data
    data = pre_process_data(data)

    training_data, testing_data, training_labels, testing_labels = train_test_split(data, labels, random_state=1)

    training_data = flatten_feature_sets(training_data)
    testing_data = flatten_feature_sets(testing_data)

    # Split data and labels into training and testing data and labels
    return training_data, testing_data, training_labels, testing_labels


def load_classifier():
    f = open(raw_input('Path to classifier: '), 'r')

    classifier = pickle.load(f)

    f.close()

    return classifier


def save_classifier(classifier):
    f = open(raw_input('Path to save classifier: '), 'wr')

    pickle.dump(classifier, f)

    f.close()
