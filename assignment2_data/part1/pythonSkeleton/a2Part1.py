import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

from NeuralNetwork import Neural_Network
from NeuralNetwork_Refined import Neural_Network_Refined


def encode_labels(labels):
    # encode 'Adelie' as 1, 'Chinstrap' as 2, 'Gentoo' as 3
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    # don't worry about this
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

    # encode 1 as [1, 0, 0], 2 as [0, 1, 0], and 3 as [0, 0, 1] (to fit with our network outputs!)
    onehot_encoder = OneHotEncoder(sparse_output=False)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    return label_encoder, integer_encoded, onehot_encoder, onehot_encoded


def perform_classification_neural_network(instances, scaler, labels, n_in, n_hidden, n_out, learning_rate,
                                          initial_hidden_layer_weights, initial_output_layer_weights):
    # We can't use strings as labels directly in the network, so need to do some transformations
    label_encoder, integer_encoded, onehot_encoder, onehot_encoded = encode_labels(labels)
    # labels = onehot_encoded

    nn = Neural_Network(n_in, n_hidden, n_out, initial_hidden_layer_weights, initial_output_layer_weights,
                        learning_rate)

    print('First instance has label {}, which is {} as an integer, and {} as a list of outputs.\n'.format(
        labels[0], integer_encoded[0], onehot_encoded[0]))

    # need to wrap it into a 2D array
    instance1_prediction = nn.predict([instances[0]])
    if instance1_prediction[0] is None:
        # This should never happen once you have implemented the feedforward.
        instance1_predicted_label = "???"
    else:
        instance1_predicted_label = label_encoder.inverse_transform(instance1_prediction)
    print('Predicted label for the first instance is: {}\n'.format(instance1_predicted_label))

    # Perform a single backpropagation pass using the first instance only. (In other words, train with 1
    #  instance for 1 epoch!). Hint: you will need to first get the weights from a forward pass.
    nn.train([instances[0]], integer_encoded, 1)

    print('Weights after performing BP for first instance only:')
    print('Hidden layer weights:\n', nn.hidden_layer_weights)
    print('Output layer weights:\n', nn.output_layer_weights)

    nn = Neural_Network(n_in, n_hidden, n_out, initial_hidden_layer_weights, initial_output_layer_weights,
                        learning_rate)
    # Train for 100 epochs, on all instances.
    nn.train(instances, integer_encoded, 100)

    print('\nAfter training:')
    print('Hidden layer weights:\n', nn.hidden_layer_weights)
    print('Output layer weights:\n', nn.output_layer_weights)

    pd_data_ts = pd.read_csv('penguins307-test.csv')
    test_labels = pd_data_ts.iloc[:, -1]
    test_instances = pd_data_ts.iloc[:, :-1]

    # scale the test according to our training data.
    test_instances = scaler.transform(test_instances)

    # Compute and print the test accuracy
    # Make predictions on test instances
    test_predictions = nn.predict(test_instances)

    # Convert test labels to one-hot encoded format
    test_labels_encoded = label_encoder.transform(test_labels)
    test_labels_onehot = onehot_encoder.transform(test_labels_encoded.reshape(-1, 1))

    # Compute accuracy
    test_accuracy = np.mean(test_predictions == np.argmax(test_labels_onehot, axis=1)) * 100
    print('Test accuracy: {:.2f}%'.format(test_accuracy))


def perform_classification_neural_network_refined(instances, scaler, labels, n_in, n_hidden, n_out, learning_rate,
                                                  initial_hidden_layer_weights, initial_output_layer_weights,
                                                  initial_hidden_bias_weights, initial_output_bias_weights):
    # We can't use strings as labels directly in the network, so need to do some transformations
    label_encoder, integer_encoded, onehot_encoder, onehot_encoded = encode_labels(labels)
    # labels = onehot_encoded

    nn = Neural_Network_Refined(n_in, n_hidden, n_out, initial_hidden_layer_weights, initial_output_layer_weights,
                                initial_hidden_bias_weights, initial_output_bias_weights, learning_rate)

    print('Weights after performing BP for first instance only:')
    print('Hidden layer weights:\n', nn.hidden_layer_weights)
    print('Output layer weights:\n', nn.output_layer_weights)

    nn = Neural_Network_Refined(n_in, n_hidden, n_out, initial_hidden_layer_weights, initial_output_layer_weights,
                                initial_hidden_bias_weights, initial_output_bias_weights, learning_rate)
    # Train for 100 epochs, on all instances.
    nn.train(instances, onehot_encoded, 100)

    print('\nAfter training:')
    print('Hidden layer weights:\n', nn.hidden_layer_weights)
    print('Output layer weights:\n', nn.output_layer_weights)

    pd_data_ts = pd.read_csv('penguins307-test.csv')
    test_labels = pd_data_ts.iloc[:, -1]
    test_instances = pd_data_ts.iloc[:, :-1]

    # scale the test according to our training data.
    test_instances = scaler.transform(test_instances)

    # Compute and print the test accuracy
    # Make predictions on test instances
    test_predictions = nn.predict(test_instances)

    # Convert test labels to one-hot encoded format
    test_labels_encoded = label_encoder.transform(test_labels)
    test_labels_onehot = onehot_encoder.transform(test_labels_encoded.reshape(-1, 1))

    # Compute accuracy
    test_accuracy = np.mean(test_predictions == np.argmax(test_labels_onehot, axis=1)) * 100
    print('Test accuracy: {:.2f}%'.format(test_accuracy))
    pass


def main():
    data = pd.read_csv('penguins307-train.csv')
    # the class label is last!
    labels = data.iloc[:, -1]
    # separate the data from the labels
    instances = data.iloc[:, :-1]
    # scale features to [0,1] to improve training
    scaler = MinMaxScaler()
    instances = scaler.fit_transform(instances)

    # Parameters. As per the handout.
    n_in = 4
    n_hidden = 2
    n_out = 3
    learning_rate = 0.2

    initial_hidden_layer_weights = np.array([[-0.28, -0.22], [0.08, 0.20], [-0.30, 0.32], [0.10, 0.01]])
    initial_output_layer_weights = np.array([[-0.29, 0.03, 0.21], [0.08, 0.13, -0.36]])

    # Initial weights for bias nodes
    initial_hidden_bias_weights = np.array([-0.02, -0.20])
    initial_output_bias_weights = np.array([-0.33, 0.26, 0.06])

    print("Neural Network without bias nodes:")
    perform_classification_neural_network(instances, scaler, labels, n_in, n_hidden, n_out, learning_rate,
                                          initial_hidden_layer_weights, initial_output_layer_weights)
    print("\n\n\n\nNeural Network with bias nodes:")
    perform_classification_neural_network_refined(instances, scaler, labels, n_in, n_hidden, n_out, learning_rate,
                                                  initial_hidden_layer_weights, initial_output_layer_weights,
                                                  initial_hidden_bias_weights, initial_output_bias_weights)



if __name__ == '__main__':
    main()
