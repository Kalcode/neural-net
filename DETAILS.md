# Neural Network Implementation for XOR, AND, OR, NAND, Half Adder, and Iris Classification Problems

This document provides a detailed breakdown of the neural network implementation used to solve XOR, AND, OR, NAND, Half Adder, and Iris Classification problems.

## Overview

The implementation consists of a `NeuralNetwork` class that creates a feedforward neural network with one hidden layer. The network is trained to learn various logical functions and classification tasks using backpropagation.

## Project Structure

- `src/NeuralNetwork.ts`: Contains the `NeuralNetwork` class implementation.
- `examples/xor/index.ts`: Example of training and testing the network on the XOR problem.
- `examples/and/index.ts`: Example of training and testing the network on the AND problem.
- `examples/or/index.ts`: Example of training and testing the network on the OR problem.
- `examples/nand/index.ts`: Example of training and testing the network on the NAND problem.
- `examples/half-adder/index.ts`: Example of training and testing the network on the Half Adder problem.
- `examples/iris/index.ts`: Example of training and testing the network on the Iris Classification problem.

## Network Architecture

- Input layer: 
  - 2 nodes for XOR, AND, OR, NAND, and Half Adder
  - 3 nodes for Iris Classification
- Hidden layer: 
  - 4 nodes for XOR and Half Adder
  - 3 nodes for AND, OR, and NAND
  - 5 nodes for Iris Classification
- Output layer: 
  - 1 node for XOR, AND, OR, and NAND
  - 2 nodes for Half Adder
  - 3 nodes for Iris Classification

Note: The number of hidden nodes can be adjusted based on the complexity of the problem.

## Half Adder Example

[... existing Half Adder explanation ...]

## Iris Classification Example

The Iris dataset is a classic machine learning problem. It involves classifying iris flowers into three species based on their sepal and petal measurements.

In our simplified version, we use three features:
1. Sepal length
2. Sepal width
3. Petal length

We classify the flowers into three species:
1. Setosa
2. Versicolor
3. Virginica

The neural network for the Iris Classification problem has:
- 3 input nodes (representing the three features)
- 5 hidden nodes
- 3 output nodes (representing the three species)

This example demonstrates the network's ability to learn a multi-class classification problem with real-valued inputs.

## Key Components

[... existing Key Components section ...]

## Detailed Breakdown

[... existing Detailed Breakdown section ...]

## Training Process

- The network is trained for 10,000 epochs in all examples.
- In each epoch, it trains on all input-output pairs for the respective problem.
- The Mean Squared Error is calculated and logged every 1000 epochs to track progress.

## Testing

After training, the network is tested on all input combinations for the respective problem to verify its learning.

## Conclusion

This implementation demonstrates a basic yet functional neural network capable of learning various logical functions and classification tasks, showcasing key concepts like feedforward propagation, backpropagation, and gradient descent optimization. The flexibility of the `NeuralNetwork` class allows it to be used for different problems by adjusting the network architecture and training data, as demonstrated by the addition of the Iris Classification example.
