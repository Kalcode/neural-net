# Neural Network Learning Project

## Overview

This project is a hands-on exploration of neural networks, implemented in TypeScript and run using Bun. The goal is to create a simple neural network from scratch to understand the underlying concepts and mechanisms.

## What is a Neural Network?

A neural network is a computational model inspired by the human brain. It consists of interconnected nodes (neurons) organized in layers. These networks can learn from data to perform tasks such as pattern recognition, classification, and prediction. Key components include:

1. Input Layer: Receives initial data
2. Hidden Layers: Process and transform data
3. Output Layer: Produces the final result
4. Weights and Biases: Adjustable parameters that the network learns
5. Activation Functions: Non-linear functions that introduce complexity
6. Training Process: Adjusting weights and biases to minimize errors

Neural networks "learn" by adjusting their internal parameters based on the difference between their predictions and actual results, a process called backpropagation.

## Features

- Implementation of a basic neural network in TypeScript
- Utilizes Bun for fast execution and development
- Designed for learning and experimentation
- Demonstrates solving various logic gate problems and the Iris classification task using neural networks

## Example Problems

This project includes several examples to demonstrate the capabilities of our neural network implementation:

1. XOR Problem: A classic non-linearly separable problem that requires a hidden layer to solve.
2. AND Gate: A simple linearly separable problem.
3. NAND Gate: The inverse of the AND gate.
4. OR Gate: Another linearly separable problem.
5. Half Adder: A combination of XOR and AND gates, demonstrating more complex logic.
6. Iris Classification: A multi-class classification problem using the famous Iris dataset.

Each of these examples showcases different aspects of neural network learning and application.

### XOR Problem

The XOR (exclusive OR) problem is a classic example used to demonstrate the capability of neural networks. The XOR function takes two binary inputs and returns 1 if exactly one of the inputs is 1, and 0 otherwise. The truth table for XOR is:

| Input 1 | Input 2 | Output |
|---------|---------|--------|
|    0    |    0    |    0   |
|    0    |    1    |    1   |
|    1    |    0    |    1   |
|    1    |    1    |    0   |

A single-layer perceptron cannot learn the XOR function, as it is not linearly separable. However, a multi-layer perceptron (MLP) with at least one hidden layer can learn to approximate the XOR function.

In this project, we train an MLP to solve the XOR problem and observe its learning progress.

### Iris Classification

The Iris classification problem is a well-known dataset in the machine learning community. It involves classifying iris flowers into three species based on four features: sepal length, sepal width, petal length, and petal width. This example demonstrates how our neural network can handle multi-class classification tasks with real-world data.

## Prerequisites

- [Bun](https://bun.sh/) installed on your system
- Basic knowledge of TypeScript and neural network concepts

## Getting Started

1. Clone this repository
2. Install dependencies:
   ```
   bun install
   ```
3. Run the project:
   ```
   bun run start
   ```

## Project Structure

(To be updated as the project progresses)

## Learning Goals

- Understand the basic architecture of neural networks
- Implement forward and backward propagation
- Experiment with different activation functions
- Learn about training processes and optimization techniques

## Future Enhancements

- Browser-based visualization of the neural network
- Implementation of different types of neural networks
- Performance optimizations

## Contributing

This project is primarily for learning purposes. However, suggestions and improvements are welcome!

## License

[MIT License](LICENSE)

## Acknowledgements

This project is inspired by various online resources and tutorials on neural networks and machine learning.
