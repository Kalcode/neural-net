# Neural Network Implementation for XOR and AND Problems

This document provides a detailed breakdown of the neural network implementation used to solve both the XOR and AND problems.

## Overview

The implementation consists of a `NeuralNetwork` class that creates a feedforward neural network with one hidden layer. The network is trained to learn either the XOR or AND function using backpropagation.

## Project Structure

- `src/NeuralNetwork.ts`: Contains the `NeuralNetwork` class implementation.
- `examples/xor/index.ts`: Example of training and testing the network on the XOR problem.
- `examples/and/index.ts`: Example of training and testing the network on the AND problem.

## Network Architecture

- Input layer: 2 nodes
- Hidden layer: 
  - 4 nodes for XOR
  - 3 nodes for AND and OR
- Output layer: 1 node

Note: The number of hidden nodes can be adjusted based on the complexity of the problem.

## Key Components

1. **Constructor**: Initializes the network structure, weights, and biases.
2. **Forward Pass**: Computes the output of the network for a given input.
3. **Backward Pass (Training)**: Updates weights and biases using backpropagation.
4. **Activation Function**: Uses the sigmoid function and its derivative.
5. **Error Calculation**: Employs Mean Squared Error (MSE) for loss computation.

## Detailed Breakdown

### 1. Network Initialization

- The `constructor` initializes the network with specified number of input, hidden, and output nodes.
- Weights are initialized randomly between -1 and 1 using `initializeWeights` method.
- Biases are also initialized randomly between -1 and 1 using `initializeBias` method.

### 2. Activation Function

- The `sigmoid` function is used as the activation function: f(x) = 1 / (1 + e^(-x))
- Its derivative, `sigmoidDerivative`, is used in backpropagation: f'(x) = f(x) * (1 - f(x))

### 3. Forward Pass

The `forward` method computes the output of the network:
- Calculates the weighted sum of inputs plus bias for each hidden node.
- Applies the sigmoid activation function to get the hidden layer output.
- Repeats the process for the output layer, using the hidden layer's output as input.

### 4. Backward Pass (Training)

The `train` method implements the backpropagation algorithm:
- Computes the output error (target - actual output).
- Calculates the gradient for the output layer.
- Propagates the error back to the hidden layer.
- Updates weights and biases for both layers using the computed gradients and a learning rate.

### 5. Error Calculation

The `meanSquaredError` method calculates the average squared difference between predicted and target values.

## Training Process

- The network is trained for 10,000 epochs in both XOR and AND examples.
- In each epoch, it trains on all four input-output pairs for the respective function.
- The Mean Squared Error is calculated and logged every 1000 epochs to track progress.

## Testing

After training, the network is tested on all four input combinations for either XOR or AND to verify its learning.

## Conclusion

This implementation demonstrates a basic yet functional neural network capable of learning both the XOR and AND functions, showcasing key concepts like feedforward propagation, backpropagation, and gradient descent optimization. The flexibility of the `NeuralNetwork` class allows it to be used for different logical operations by adjusting the network architecture and training data.
