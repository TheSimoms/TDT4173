from sklearn import svm, neighbors
from sklearn.metrics import classification_report, confusion_matrix

from utils import (prepare_data, extract_windows, load_arbitrary_image, index_to_letter, load_classifier,
                   save_classifier, pre_process_data, flatten_feature_sets, draw_image_with_windows)


def analysis():
    print('Preparing data...')

    # Create a classifier: support vector classifier
    training_data, testing_data, training_labels, testing_labels = prepare_data()

    classifiers = (
        (svm.SVC(kernel='poly', degree=3, probability=True), 'Support vector machine'),
        (neighbors.KNeighborsClassifier(n_neighbors=8, weights='distance'), 'K-nearest neighbours')
    )

    print('Ready to classify')

    for classifier in classifiers:
        print('\nClassifier: %s' % classifier[1])

        # Build the pipeline
        classifier = classifier[0]

        print('Training...')

        # Train the classifier using the training data
        classifier.fit(training_data, training_labels)

        print('Training complete')
        print('Testing...')

        predictions = classifier.predict(testing_data)

        print('Testing complete')

        # Print the prediction results
        print('Results:')
        print(confusion_matrix(testing_labels, predictions))
        print(classification_report(testing_labels, predictions))


def classify_windows(classifier, windows):
    results = []

    # Predict the windows
    probabilities = classifier.predict_proba(windows)

    # Maximum probability calculated
    max_probability = -1

    # Most likely label
    max_probability_label = None

    # Index of most likely window
    max_probability_window = None

    # Iterate all windows, find most likely label for each window
    for i in range(len(probabilities)):
        window_probability = probabilities[i]

        # Maximum probability calculated
        probability = max(window_probability)

        # Most likely label
        probability_index = window_probability.tolist().index(probability)

        # Add probability and index to results
        results.append((probability, probability_index))

        # Check if probability is greater than previously found maximum
        if probability > max_probability:
            # Update maximum variables
            max_probability = probability
            max_probability_label = probability_index
            max_probability_window = i

    # Return results, and maximum likely label
    return results, (max_probability, index_to_letter(max_probability_label), max_probability_window)


def new_classifier():
    print('Preparing data...')

    training_data, testing_data, training_labels, testing_labels = prepare_data()

    classifier = svm.SVC(kernel='poly', degree=3, probability=True)

    print('Training classifier...')

    classifier.fit(training_data, training_labels)

    return classifier, training_data, testing_data, training_labels, testing_labels


def detect_arbitrary_image(image_path, classifier):
    print('Extracting windows...')

    # Extract and flatten windows
    windows, window_positions = extract_windows(load_arbitrary_image(image_path))
    windows = flatten_feature_sets(pre_process_data(windows))

    # Set up classifier
    if classifier is None:
        classifier, training_data, testing_data, training_labels, testing_labels = new_classifier()
        classifier.fit(training_data + testing_data, training_labels + testing_labels)

    print('Detecting letters from windows...')

    # Classify the windows
    results, (max_probability, max_probability_letter, max_probability_window) = classify_windows(classifier, windows)

    # Draw the most likely classification
    draw_image_with_windows(image_path, [window_positions[max_probability_window]])

    # Print probability and classified label
    print('Classified letter: %s, with probability %f' % (max_probability_letter, max_probability))

if input('Analysis? '):
    analysis()
elif input('Detect? '):
    if input('Load classifier model? '):
        cls = load_classifier()
    else:
        cls = None

    detect_arbitrary_image(raw_input('Path to image: '), cls)
else:
    cls, _, _, _, _ = new_classifier()

    save_classifier(cls)
