# Name: Rhea D'Souza

import numpy as np
import pandas as pd
import sys


def step_function(x):
    return np.where(x > 0, 1, 0)

class Perceptron:
    def __init__(self, learning_rate=1, n_iters=100):
        self.lr = learning_rate
        self.n_iters = max(n_iters, 100)
        self.activation_func = step_function

        self.weights = None  # initialised in the fit function.
        self.bias = None  # No bias as the dummy feature acts as our bias
        self.iterations_to_convergence = None
        self.misclassified_instances = None

    def fit(self, X, y_train):
        """
        Until the perceptron is always right (or some limit):
        present an example (+ve or -ive)
        If perceptron is correct, do nothing
        Else if -ve example and wrong:
            (i.e. weights on active features are too high)
            Subtract feature vector from weight vector
        Else if +ive example and wrong:
            (i.e. weights on active features are too low)
            Add feature vector to weight vector
        :return:
        """
        X_train = np.array(X)  # Converts pd dataframe into a np array
        n_samples, n_features = X_train.shape

        # Initialise params
        self.weights = np.random.uniform(-1, 1, size=n_features)
        self.weights[0] = 1
        y_ = np.where(y_train > 0, 1, 0)

        iteration = 0
        self.iterations_to_convergence = 0
        while iteration < self.n_iters:
            misclassified = 0
            for i, x_i in enumerate(X_train):
                weighted_sum = np.dot(x_i, self.weights)
                y_pred = self.activation_func(weighted_sum)

                # Perceptron update rule
                update = self.lr * (y_[i] - y_pred)

                if update != 0:
                    misclassified += 1
                    self.weights += update * x_i

            if misclassified == 0:
                self.iterations_to_convergence = iteration + 1
                self.misclassified_instances = 0
                break

            self.misclassified_instances = misclassified
            iteration += 1


    def get_weights(self):
        return self.weights

    def predict(self, X):
        output = np.dot(X, self.weights)
        y_pred = self.activation_func(output)
        return y_pred


def load_dataset(filename) -> pd.DataFrame:
    data = pd.read_csv(filename, delim_whitespace=True)
    data.insert(0, 'f0', 1)  # Inserts the 'dummy feature' into the data with all 1s

    print("Shape of data:", data.shape)
    print("Columns of data:", data.columns)
    print(data.head())
    return data


def get_labels_and_data(dataset):
    labels = dataset['class']
    data = dataset.drop(columns='class')
    return labels, data


def binary_encode_labels(labels):
    return np.where(labels == 'g', 1, 0)


def train_test_split(data, labels, test_ratio=0.3):
    len_data = len(data)
    shuffled_idxs = np.random.permutation(len_data)

    num_test_samples = int(len_data * test_ratio)
    test_indices = shuffled_idxs[:num_test_samples]
    train_indices = shuffled_idxs[num_test_samples:]

    train_data = data.iloc[train_indices].to_numpy()
    train_labels = labels[train_indices]
    test_data = data.iloc[test_indices].to_numpy()
    test_labels = labels[test_indices]

    return train_data, train_labels, test_data, test_labels


def perform_classification(X, y, X_test, y_test, lr=1.0, iters=100):
    perceptron = Perceptron(learning_rate=lr, n_iters=iters)
    perceptron.fit(X, y)
    y_pred = perceptron.predict(X_test)

    # Print out a receipt of the classification's training and test information
    final_weights = f'Final Weights:\n{perceptron.get_weights()}'
    iter_to_converge = f'Iterations to convergence: {perceptron.iterations_to_convergence}'
    misclassified_training = f'Misclassified instances - Training: {perceptron.misclassified_instances}'
    misclassified_test = f'Misclassified instances - Testing: {np.sum(y_pred != y_test)}'
    accuracy = f'Accuracy: {np.mean(y_test == y_pred) * 100:.2f}%'

    print(iter_to_converge, '\n', final_weights,
          '\n', misclassified_training, '\n', misclassified_test,
          '\n', accuracy)


def main(file_name):
    dataset = load_dataset(file_name)
    test_ratio = 0.3

    print('\n\n Training Data == Test Data\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    labels, data = get_labels_and_data(dataset)
    labels_encoded = binary_encode_labels(labels)
    perform_classification(data, labels_encoded, data, labels_encoded, lr=0.01, iters=160)

    print(f'\n\n Training Data: {100-test_ratio*100:.0f}% and Test Data: {test_ratio*100:.0f}%\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    X_train, y_train, X_test, y_test = train_test_split(data, labels_encoded, test_ratio=test_ratio)
    perform_classification(X_train, y_train, X_test, y_test, lr=0.01, iters=320)



if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python perceptron.py <data_file>")
        sys.exit(1)

    file_name = sys.argv[1]
    main(file_name)
