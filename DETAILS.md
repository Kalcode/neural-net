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
  - 4 nodes for Iris Classification
- Hidden layer: 
  - 4 nodes for XOR and Half Adder
  - 3 nodes for AND, OR, and NAND
  - 8 nodes for Iris Classification
- Output layer: 
  - 1 node for XOR, AND, OR, and NAND
  - 2 nodes for Half Adder
  - 3 nodes for Iris Classification

Note: The number of hidden nodes can be adjusted based on the complexity of the problem.

## XOR Problem

The XOR (exclusive OR) problem is a classic example used to demonstrate the capability of neural networks. The XOR function takes two binary inputs and returns 1 if exactly one of the inputs is 1, and 0 otherwise. The truth table for XOR is:

| Input 1 | Input 2 | Output |
|---------|---------|--------|
|    0    |    0    |    0   |
|    0    |    1    |    1   |
|    1    |    0    |    1   |
|    1    |    1    |    0   |

A single-layer perceptron cannot learn the XOR function, as it is not linearly separable. However, a multi-layer perceptron (MLP) with at least one hidden layer can learn to approximate the XOR function.

## AND Problem

The AND problem is a simple logical operation where the output is 1 only if both inputs are 1. Unlike XOR, it is linearly separable and can be learned by a single-layer perceptron. The truth table for AND is:

| Input 1 | Input 2 | Output |
|---------|---------|--------|
|    0    |    0    |    0   |
|    0    |    1    |    0   |
|    1    |    0    |    0   |
|    1    |    1    |    1   |

## OR Problem

The OR problem is another simple logical operation where the output is 1 if at least one of the inputs is 1. Like AND, it is linearly separable. The truth table for OR is:

| Input 1 | Input 2 | Output |
|---------|---------|--------|
|    0    |    0    |    0   |
|    0    |    1    |    1   |
|    1    |    0    |    1   |
|    1    |    1    |    1   |

## NAND Problem

The NAND (NOT AND) problem is the negation of the AND operation. The output is 0 only if both inputs are 1. It is also linearly separable. The truth table for NAND is:

| Input 1 | Input 2 | Output |
|---------|---------|--------|
|    0    |    0    |    1   |
|    0    |    1    |    1   |
|    1    |    0    |    1   |
|    1    |    1    |    0   |

## Half Adder Example

The Half Adder is a digital circuit that performs addition of two binary digits. It produces two outputs: Sum and Carry.

- Sum: The result of adding two bits (modulo 2)
- Carry: The overflow bit when adding two bits

Truth Table for Half Adder:
| Input 1 | Input 2 | Sum | Carry |
|---------|---------|-----|-------|
|    0    |    0    |  0  |   0   |
|    0    |    1    |  1  |   0   |
|    1    |    0    |  1  |   0   |
|    1    |    1    |  0  |   1   |

The neural network for the Half Adder problem has:
- 2 input nodes (representing the two input bits)
- 4 hidden nodes
- 2 output nodes (representing Sum and Carry)

This example demonstrates the network's ability to learn more complex logical operations with multiple outputs.

## Iris Classification Example

The Iris classification problem is a well-known dataset in the machine learning community. It involves classifying iris flowers into three species based on four features: sepal length, sepal width, petal length, and petal width. This example demonstrates how our neural network can handle multi-class classification tasks with real-world data.

We use all four features:
1. Sepal length
2. Sepal width
3. Petal length
4. Petal width

We classify the flowers into three species:
1. Setosa
2. Versicolor
3. Virginica

The neural network for the Iris Classification problem has:
- 4 input nodes (representing the four features)
- 8 hidden nodes
- 3 output nodes (representing the three species)

This example demonstrates the network's ability to learn a multi-class classification problem with real-valued inputs.

### Sample Output of the model's internal weights and biases
Here is an image where I console logged the various data inside the Neural Network instance

![image](images/sample_nn_output.png)

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

- The network is trained for a specified number of epochs in all examples (typically 10,000 or more).
- In each epoch, it trains on all input-output pairs for the respective problem.
- The Mean Squared Error is calculated and logged periodically to track progress.

## Testing

After training, the network is tested on all input combinations for the respective problem to verify its learning.

## Conclusion

This implementation demonstrates a basic yet functional neural network capable of learning various logical functions and classification tasks, showcasing key concepts like feedforward propagation, backpropagation, and gradient descent optimization. The flexibility of the `NeuralNetwork` class allows it to be used for different problems by adjusting the network architecture and training data, as demonstrated by the addition of the Iris Classification example.
