import numpy as np
class Neural_Network_Refined:
    # Initialize the network
    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights, output_layer_weights,
                 hidden_bias_weights, output_bias_weights, learning_rate):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        self.hidden_layer_weights = hidden_layer_weights
        self.output_layer_weights = output_layer_weights
        self.hidden_bias_weights = hidden_bias_weights
        self.output_bias_weights = output_bias_weights

        self.learning_rate = learning_rate

    # Calculate neuron activation for an input
    def sigmoid(self, input):
        output = 1 / (1 + np.exp(-input))
        return output

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # Feed forward pass input to a network output
    def forward_pass(self, inputs):
        hidden_layer_outputs = []
        for i in range(self.num_hidden):
            # Calculate the weighted sum for the hidden layer neuron i
            weighted_sum = np.dot(inputs, self.hidden_layer_weights[:, i]) + self.hidden_bias_weights[i]
            # Apply the sigmoid activation function
            output = self.sigmoid(weighted_sum)
            hidden_layer_outputs.append(output)

        # Convert the hidden layer outputs to a numpy array for consistency
        hidden_layer_outputs = np.array(hidden_layer_outputs)

        output_layer_outputs = []
        for i in range(self.num_outputs):
            # Calculate the weighted sum for the output layer neuron i
            weighted_sum = np.dot(hidden_layer_outputs, self.output_layer_weights[:, i]) + self.output_bias_weights[i]
            # Apply the sigmoid activation function
            output = self.sigmoid(weighted_sum)
            output_layer_outputs.append(output)

        # Convert the output layer outputs to a numpy array for consistency
        output_layer_outputs = np.array(output_layer_outputs)

        return hidden_layer_outputs, output_layer_outputs

    # Backpropagation error and store in neurons
    def backward_propagate_error(self, inputs, hidden_layer_outputs, output_layer_outputs, desired_outputs):
        # Calculate output layer betas.
        output_layer_betas = (output_layer_outputs - desired_outputs) * self.sigmoid_derivative(output_layer_outputs)

        # Calculate hidden layer betas.
        hidden_layer_betas = np.dot(output_layer_betas, self.output_layer_weights.T) * self.sigmoid_derivative(
            hidden_layer_outputs)

        # This is a HxO array (H hidden nodes, O outputs)
        delta_output_layer_weights = np.outer(hidden_layer_outputs, output_layer_betas)

        # This is a IxH array (I inputs, H hidden nodes)
        delta_hidden_layer_weights = np.outer(inputs, hidden_layer_betas)

        # Return the weights we calculated, so they can be used to update all the weights.
        return delta_output_layer_weights, delta_hidden_layer_weights

    def update_weights(self, delta_output_layer_weights, delta_hidden_layer_weights):
        self.output_layer_weights += self.learning_rate * delta_output_layer_weights
        self.hidden_layer_weights += self.learning_rate * delta_hidden_layer_weights

    def train(self, instances, desired_outputs, epochs):
        for epoch in range(epochs):
            total_loss = 0
            predictions = []
            for i, instance in enumerate(instances):
                hidden_layer_outputs, output_layer_outputs = self.forward_pass(instance)
                delta_output_layer_weights, delta_hidden_layer_weights, = self.backward_propagate_error(
                    instance, hidden_layer_outputs, output_layer_outputs, desired_outputs[i])
                predicted_class = np.argmax(output_layer_outputs)
                predictions.append(predicted_class)

                loss = np.square(np.subtract(output_layer_outputs, desired_outputs[i])).mean()
                total_loss += loss

                # We use online learning, i.e. update the weights after every instance.
                self.update_weights(delta_output_layer_weights, delta_hidden_layer_weights)

            # Print accuracy achieved over this epoch
            acc = f'{np.mean(predictions == np.argmax(desired_outputs, axis=1)) * 100:.2f}%'
            print('Epoch:', epoch + 1, 'Accuracy:', acc)

    def predict(self, instances):
        predictions = []
        for instance in instances:
            hidden_layer_outputs, output_layer_outputs = self.forward_pass(instance)
            predicted_class = np.argmax(output_layer_outputs)  # Should be 0, 1, or 2.
            predictions.append(predicted_class)
        return predictions
