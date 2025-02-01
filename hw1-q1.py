#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import numpy as np
import matplotlib.pyplot as plt

import time
import utils

class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        Q1.1 (a)
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        learning_rate = kwargs.get('learning_rate', 0.001)

        # Predict the class label
        predicted_label = np.argmax(self.W.dot(x_i))

        # Update the weights if the prediction is incorrect
        if predicted_label != y_i:
            # Increase the weight of the correct class
            self.W[y_i] += learning_rate * x_i
            # Decrease the weight of the predicted incorrect class
            self.W[predicted_label] -= learning_rate * x_i


class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001, l2_penalty=0.0, **kwargs):
        """
        Q1.2 (a,b)
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # Get the label scores (n_labels x 1).
        label_scores = np.expand_dims(self.W.dot(x_i), axis=1)

        # One-hot encode the true label (n_labels x 1).
        y_one_hot = np.zeros((np.size(self.W, 0), 1))
        y_one_hot[y_i] = 1

        # Apply the softmax function to calculate the probabilities (n_labels x 1).
        exp_scores = np.exp(label_scores - np.max(label_scores))
        label_probabilities = exp_scores / np.sum(exp_scores)

        # Compute the gradient (n_labels, n_features).
        gradient = (y_one_hot - label_probabilities).dot(np.expand_dims(x_i, axis=1).T)
        if l2_penalty > 0:
            gradient += l2_penalty * self.W

        # Update the weights using SGD.
        self.W += learning_rate * gradient


class MLP(object):
    def __init__(self, n_classes, n_features, hidden_size):
        # Initialize an MLP with multiple layers
        units = [n_features, hidden_size, n_classes]

        # Initialize the weights (W) and biases (b) for all layers
        self.weights = []
        self.biases = []

        for i in range(len(units) - 1):
            # Print the current layer being initialized
            print(f"Initializing layer {i} with input size {units[i]} and output size {units[i + 1]}")

            # Initialize the weights for layer i (units[i] -> units[i+1])
            # Weights are drawn from a normal distribution with mean=0.1 and variance=0.1
            self.weights.append(np.random.normal(loc=0.1, scale=0.1, size=(units[i + 1], units[i])))

            # Print the shape of the weights matrix for the current layer
            print(f"Weights for layer {i}: {self.weights[-1].shape}")

            # Initialize the biases for layer i (size: units[i+1],)
            # Biases are initialized to zero
            self.biases.append(np.zeros((units[i + 1], 1)))

            # Print the shape of the bias vector for the current layer
            print(f"Biases for layer {i}: {self.biases[-1].shape}")

        print("All weights:")
        for i, weight in enumerate(self.weights):
            print(f"Layer {i} weight shape: {weight.shape}")

        print("All biases:")
        for i, bias in enumerate(self.biases):
            print(f"Layer {i} bias shape: {bias.shape}")

        # Assign the number of input features and output classes
        self.features = n_features
        self.n_classes = n_classes

    def softmax(self, x):
        # Subtract max for numerical stability
        x_max = np.max(x, axis=0, keepdims=True)

        # Exponentiate the adjusted values
        exp_x = np.exp(x - x_max)

        # Normalize by the sum of exponentials to get probabilities
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)

    def forward(self, x):
        # Number of layers based on weights list length
        num_layers = len(self.weights)

        # ReLU activation function
        g = lambda z: np.maximum(0, z)

        hiddens, z = [], 0

        # Compute hidden layers
        for i in range(num_layers):
            # Reshape input for first layer, otherwise use previous hidden output
            h = x.reshape(-1, 1) if i == 0 else hiddens[i - 1]

            # Pre-activation: z = W * h + b
            z = self.weights[i].dot(h) + self.biases[i]

            # Apply ReLU for hidden layers, not for output layer
            if i < num_layers - 1:
                hiddens.append(g(z))

        # Output layer has no activation
        output = z

        return output, hiddens


    def backward(self, x, y, output, hiddens):
        # Number of layers
        num_layers = len(self.weights)

        # Compute softmax output and gradient of the loss w.r.t. output
        probs = self.softmax(output)
        grad_z = probs - y.reshape(-1, 1)

        # Initialize lists for gradients
        grad_weights = []
        grad_biases = []

        # Backpropagate the gradient
        for i in range(num_layers - 1, -1, -1):
            # Get input for current layer
            h = x.reshape(-1, 1) if i == 0 else hiddens[i - 1]

            # Compute gradients for weights and biases
            grad_weights.append(np.dot(grad_z, h.T))
            grad_biases.append(grad_z)

            # Compute gradient for the next layer
            grad_h = self.weights[i].T.dot(grad_z)
            grad_z = grad_h * (h > 0)  # ReLU derivative

        # Reverse the gradients to match layer order
        grad_weights.reverse()
        grad_biases.reverse()

        # Return gradients
        return grad_weights, grad_biases

    def compute_loss(self, output, y):
        # Add epsilon to prevent log(0) errors
        epsilon = 1e-15
        probs = self.softmax(output)
        # Calculate the negative log-likelihood loss
        loss = -np.sum(y * np.log(np.clip(probs, epsilon, 1 - epsilon)))
        return loss

    def predict(self, X):
        predicted_labels = []
        for x in X:
            # Perform forward pass and get the class with the highest probability
            output, _ = self.forward(x)
            y_hat = np.argmax(self.softmax(output))
            predicted_labels.append(y_hat)

        # Convert list to numpy array
        predicted_labels = np.array(predicted_labels)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]

        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001, **kwargs):
        num_layers = len(self.weights)
        total_loss = 0

        # Iterate through each observation and label
        for x, label in zip(X, y):
            # Convert label to one-hot encoding
            y_one_hot = np.zeros(self.n_classes)
            y_one_hot[label] = 1

            # Forward pass
            output, hiddens = self.forward(x)

            # Compute and accumulate loss
            loss = self.compute_loss(output, y_one_hot)
            total_loss += loss

            # Backpropagation to compute gradients
            grad_weights, grad_biases = self.backward(x, y_one_hot, output, hiddens)

            # Update weights and biases using gradients
            for i in range(num_layers):
                self.weights[i] -= learning_rate * grad_weights[i]
                self.biases[i] -= learning_rate * grad_biases[i]

        return total_loss


def plot(epochs, train_accs, val_accs, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, val_accs, label='validation')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

def plot_loss(epochs, loss, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='train')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

def plot_w_norm(epochs, w_norms, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('W Norm')
    plt.plot(epochs, w_norms, label='train')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=100,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    parser.add_argument('-l2_penalty', type=float, default=0.0,)
    parser.add_argument('-data_path', type=str, default='datasets/intel_landscapes_v2.npz',)
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_dataset(data_path=opt.data_path, bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]
    n_classes = np.unique(train_y).size
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    train_loss = []
    weight_norms = []
    valid_accs = []
    train_accs = []

    start = time.time()

    print('initial train acc: {:.4f} | initial val acc: {:.4f}'.format(
        model.evaluate(train_X, train_y), model.evaluate(dev_X, dev_y)
    ))

    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        if opt.model == 'mlp':
            loss = model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        else:
            model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate,
                l2_penalty=opt.l2_penalty,
            )
        
        train_accs.append(model.evaluate(train_X, train_y))
        valid_accs.append(model.evaluate(dev_X, dev_y))
        if opt.model == 'mlp':
            print('loss: {:.4f} | train acc: {:.4f} | val acc: {:.4f}'.format(
                loss, train_accs[-1], valid_accs[-1],
            ))
            train_loss.append(loss)
        elif opt.model == "logistic_regression":
            weight_norm = np.linalg.norm(model.W)
            print('train acc: {:.4f} | val acc: {:.4f} | W norm: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1], weight_norm,
            ))
            weight_norms.append(weight_norm)
        else:
            print('train acc: {:.4f} | val acc: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1],
            ))
    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print('Training took {} minutes and {} seconds'.format(minutes, seconds))
    print('Final test acc: {:.4f}'.format(
        model.evaluate(test_X, test_y)
        ))

    # plot
    plot(epochs, train_accs, valid_accs, filename=f"images/1/Q1-{opt.model}-accs.png")
    if opt.model == 'mlp':
        plot_loss(epochs, train_loss, filename=f"images/1/Q1-{opt.model}-loss.png")
    elif opt.model == 'logistic_regression':
        plot_w_norm(epochs, weight_norms, filename=f"images/1/Q1-{opt.model}-w_norms.png")
    with open(f"images/Q1-{opt.model}-results.txt", "w") as f:
        f.write(f"Final test acc: {model.evaluate(test_X, test_y)}\n")
        f.write(f"Training time: {minutes} minutes and {seconds} seconds\n")


if __name__ == '__main__':
    main()

