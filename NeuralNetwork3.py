import math
import sys
import time

import numpy as np

class Sigmoid:
    @staticmethod
    def sigmoid(s):
        return 1 / (1 + np.math.exp((-1) * s))

    @staticmethod
    def sigmoid_derivative(X):
        return X * (1 - X)


class NeuralNetwork:
    # 'layers' is a numpy array of number of perceptrons in each layer identified by the array index
    # X is required to get the size of input for the first NNLayer's d_prev value
    def __init__(self, input_size, hidden_layers_sizes):
        self.layer_sizes = np.array(hidden_layers_sizes)
        self.data_size = np.array([input_size])
        # print(self.data_size)
        self.layer_sizes = np.append(self.data_size, self.layer_sizes)  # We now have Input + Hidden layers in the array
        assert len(self.layer_sizes) > 1
        # print(self.layer_list)
        self._initialize_layers()

    def _initialize_layers(self):
        self.layers = list()  # This is the list of NNLayers
        for index in range(len(self.layer_sizes) - 1):
            # print(self.layer_list[index + 1])
            # print(self.layer_list[index])
            layer = NNLayer(self.layer_sizes[index + 1], self.layer_sizes[index])
            self.layers.append(layer)
        # final output layer with 10 perceptrons, for nums 0 to 9
        output_layer = NNLayer(10, self.layer_sizes[-1])
        self.layers.append(output_layer)

        # Debugging
        assert len(self.layers) == 2

    def feed_forward(self, X):
        for level, layer in enumerate(self.layers):
            if level == 0:
                layer.feed_forward_sigmoid(X)
            elif level < len(self.layers)-1:
                layer.feed_forward_sigmoid(self.layers[level - 1].outputs)
            else:  # For the last layer apply the softmax function
                layer.feed_forward_softmax(self.layers[level - 1].outputs)

    def back_propagation(self):
        # Base case is the output layer
        output_layer = self.layers[-1]
        for i, perceptron in enumerate(output_layer.perceptrons):
            y = self.label == i  # index of the perceptron 0 through 9, must match the label (number)
            delta = 2 * (perceptron.output - y) * (Sigmoid.sigmoid_derivative(perceptron.output))
            perceptron.update_delta(delta)

        # Start from hidden layer that comes before output layer, up to 0th element which is the first hidden layer
        # in the list. In the range below, -1 is exclusive, iteration goes on until layers[0]
        for level in range(len(self.layers) - 2, -1, -1):
            hidden_layer = self.layers[level]
            for i, perceptron in enumerate(hidden_layer.perceptrons):
                delta_j = list()
                w_ij = list()
                for p in self.layers[level + 1].perceptrons:
                    delta_j.append(p.delta)
                    w_ij.append(p.weights[i])

                delta = Sigmoid.sigmoid_derivative(perceptron.output) * np.dot(np.array(w_ij), np.array(delta_j))
                perceptron.update_delta(delta)

    def update_weights(self, lr):
        for layer in self.layers:
            layer.update_weights(lr)

    def train(self, X, label, lr=0.1):
        self.label = label
        self.feed_forward(X)
        self.back_propagation()
        self.update_weights(lr)

    # predict which number from 0 to 9 the input is most likely to be
    def predict(self, X):
        self.feed_forward(X)
        output_layer = self.layers[-1]

        return np.argmax(output_layer.outputs)


# Initially d_prev would be the number of pixels
class NNLayer:
    def __init__(self, d, d_prev):
        # weights w_ij correspond to all of last layers i elements connnected with current j element weights
        self.perceptrons = self._initialize_perceptrons(d, d_prev)
        self.outputs = np.zeros(d)

    def _initialize_perceptrons(self, d, d_prev):
        perceptron_list = []
        # all layers start with 1 for the bias
        # create from 0 to d+1 exclusive, perceptrons with randomized weights
        for i in range(d):
            # Random non-zero weight initialization
            weights = np.random.uniform(-0.01, 0.01, d_prev + 1)
            #             print("perceptron " + str(i) + " weights are: " + str(weights))
            perceptron_list.append(Perceptron(weights=weights))
        #         print("")
        return perceptron_list

    def feed_forward_sigmoid(self, X):
        padded_X = np.array([1])
        padded_X = np.append(padded_X, X)
        for i in range(len(self.outputs)):
            self.perceptrons[i].generate_output_sigmoid(padded_X)
            #             print("perceptron " + str(i) + " output is: " + str(self.percpetrons[i].output))
            self.outputs[i] = self.perceptrons[i].output

    def feed_forward_softmax(self, X):
        padded_X = np.array([1])
        padded_X = np.append(padded_X, X)

        exponents_sum = 0.00000000000001  # non-zero value to avoid divide by zero
        for i in range(len(self.outputs)):
            self.perceptrons[i].generate_output_softmax(padded_X)
            exponents_sum += self.perceptrons[i].output

        for i in range(len(self.outputs)):
            self.perceptrons[i].output /= exponents_sum
            self.outputs[i] = self.perceptrons[i].output

    def update_weights(self, lr):
        for p in self.perceptrons:
            p.update_weights(lr)


class Perceptron:
    def __init__(self, output=1, weights=None):
        self.output = output
        self.weights = weights
        self.delta = 0

    def update_weights(self, lr):
        self.weights -= lr * (self.X * self.delta)

    def generate_output_sigmoid(self, X):
        self.X = X
        self._apply_sigmoid()

    # Will only apply exponent, outside need to divide by their sum to normalize
    def generate_output_softmax(self, X):
        self.X = X
        self._apply_exponent()

    def summation_phase(self):
        return np.dot(self.weights, self.X)

    def _apply_sigmoid(self):
        self.output = Sigmoid.sigmoid(self.summation_phase())

    def _apply_exponent(self):
        self.output = math.exp(self.summation_phase())

    def update_delta(self, delta):
        self.delta = delta


#  run program as python3 NeuralNetwork.py train_image.csv train_label.csv, or it can be run as python3 NeuralNetwork.py
def main():
    start_time = time.time()

    # Each file will contain at least 1 image and at most 60,000 images and at the least 10,000 using MNIST data
    # Default file names if no cli input is given
    train_image_fname = "train_image.csv"
    train_label_fname = "train_label.csv"
    test_image_fname = "test_image.csv"
    test_label_fname = "test_label.csv"  # File name for output

    # If 4 input arguments are given (as in python3 NeuralNetwork3.py train_image.csv train_label.csv test_image.csv
    if len(sys.argv) == 4:
        train_image_fname = sys.argv[1]
        train_label_fname = sys.argv[2]
        test_image_fname = sys.argv[3]

    train_images = np.genfromtxt(train_image_fname, delimiter=',')
    train_images = train_images/255  # normalize by dividing by the max 8 bit value
    train_labels = np.genfromtxt(train_label_fname, delimiter=',')

    #  TODO: may need to adjust epochs based on how many images are in the input
    EPOCHS = 5
    LEARNING_RATE = 0.1
    HIDDEN_LAYER_PERCEPTRONS_NUMBER = 100
    N = len(train_images)  # number of images

    nn = NeuralNetwork(len(train_images[0]), [HIDDEN_LAYER_PERCEPTRONS_NUMBER])
    for epoch in range(EPOCHS):
        for n in range(N):
            index = np.random.randint(0, N)
            image = train_images[index]
            nn.train(image, train_labels[index], LEARNING_RATE)

    # # Train Data Testing
    # correct_predictions = 0
    # total_count = 0
    # for index, image in enumerate(train_images):
    #     total_count += 1
    #     prediction = nn.predict(image)
    #     correct_predictions += 1 if prediction == train_labels[index] else 0
    #     print("image " + str(index) + ": " + str(prediction))
    # print("Accuracy: " + str(float(correct_predictions) / float(total_count)))

    # Test data Testing
    test_images = np.genfromtxt(test_image_fname, delimiter=',')
    test_images = test_images / 255  # normalize by dividing by the max 8 bit value
    test_labels = np.genfromtxt(test_label_fname, delimiter=',')

    correct_predictions = 0
    total_count = 0
    for index, image in enumerate(test_images):
        total_count += 1
        prediction = nn.predict(image)
        correct_predictions += 1 if prediction == test_labels[index] else 0
        # print("image " + str(index) + ": " + str(prediction))
    print("Accuracy: " + str(float(correct_predictions)/float(total_count)))

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()
