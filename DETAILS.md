# Neural Network Implementation for XOR, AND, OR, NAND, and Half Adder Problems

This document provides a detailed breakdown of the neural network implementation used to solve XOR, AND, OR, NAND, and Half Adder problems.

## Overview

The implementation consists of a `NeuralNetwork` class that creates a feedforward neural network with one hidden layer. The network is trained to learn various logical functions using backpropagation.

## Project Structure

- `src/NeuralNetwork.ts`: Contains the `NeuralNetwork` class implementation.
- `examples/xor/index.ts`: Example of training and testing the network on the XOR problem.
- `examples/and/index.ts`: Example of training and testing the network on the AND problem.
- `examples/or/index.ts`: Example of training and testing the network on the OR problem.
- `examples/nand/index.ts`: Example of training and testing the network on the NAND problem.
- `examples/half-adder/index.ts`: Example of training and testing the network on the Half Adder problem.

## Network Architecture

- Input layer: 2 nodes
- Hidden layer: 
  - 4 nodes for XOR and Half Adder
  - 3 nodes for AND, OR, and NAND
- Output layer: 
  - 1 node for XOR, AND, OR, and NAND
  - 2 nodes for Half Adder

Note: The number of hidden nodes can be adjusted based on the complexity of the problem.

## Half Adder Example

The Half Adder is a digital circuit that performs addition of two binary digits. It produces two outputs: Sum and Carry.

- Sum: The result of adding two bits (modulo 2)
- Carry: The overflow bit when adding two bits

Truth Table for Half Adder:
- Input: 0, 0 -> Output: Sum = 0, Carry = 0
- Input: 0, 1 -> Output: Sum = 1, Carry = 0
- Input: 1, 0 -> Output: Sum = 1, Carry = 0
- Input: 1, 1 -> Output: Sum = 0, Carry = 1

The neural network for the Half Adder problem has:
- 2 input nodes (representing the two input bits)
- 4 hidden nodes
- 2 output nodes (representing Sum and Carry)

This example demonstrates the network's ability to learn more complex logical operations with multiple outputs.

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
