from utils import load_data, pre_process_data, sliding_windows

from sklearn import svm, neighbors
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from skimage.color import rgb2gray


# Load data and labels
data, labels = load_data()

# Pre process data
data = pre_process_data(data)

# Split data and labels into training and testing data and labels
training_data, testing_data, training_labels, testing_labels = train_test_split(data, labels, random_state=1)

# flattening training data
training_data = [data.flatten() for data in training_data]

# Create a classifier: support vector classifier
classifiers = (
    (neighbors.KNeighborsClassifier(weights='distance'), 'K-nearest neighbours'),
    (svm.SVC(kernel='poly', degree=3, probability=True), 'Support vector machine'),
)

print('Ready to classify')

for classifier in classifiers:
    print('\nClassifier: %s' % classifier[1])

    # Build the pipeline
    classifier = classifier[0]

    print('Training')

    # Train the classifier using the training data
    classifier.fit(training_data, training_labels)

    print('Training complete')
    print('Testing')

    testing_data = [data.flatten() for data in testing_data]

    predictions = classifier.predict(testing_data)

    print('Testing complete')

    # Print the prediction results
    print('Results:')
    print(classification_report(testing_labels, predictions))
    print(confusion_matrix(testing_labels, predictions))
