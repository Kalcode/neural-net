# Neuron Implementation

The `Neuron` class represents a single neuron in our neural network. It's implemented in the `src/Neuron.ts` file.

## Class Structure

```typescript
export class Neuron {
    private weights: number[];
    private bias: number;

    constructor(inputSize: number) {
        // ...
    }

    private sigmoid(x: number): number {
        // ...
    }

    forward(inputs: number[]): number {
        // ...
    }
}
```

## Key Components

1. **Weights**: Each neuron has a set of weights, one for each input it receives. These are initialized randomly between -1 and 1.

2. **Bias**: The bias is an additional parameter that allows the neuron to fit the data better. It's also initialized randomly between -1 and 1.

3. **Activation Function**: We use the sigmoid function as our activation function. It maps any input to a value between 0 and 1.

4. **Forward Pass**: The `forward` method calculates the neuron's output for a given set of inputs.

## Implementation Details

### Constructor

```typescript
constructor(inputSize: number) {
    this.weights = Array.from({ length: inputSize }, () => Math.random() * 2 - 1);
    this.bias = Math.random() * 2 - 1;
}
```

The constructor initializes the weights and bias randomly. The number of weights is determined by the `inputSize` parameter.

### Sigmoid Function

```typescript
private sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
}
```

This is the activation function used by the neuron. It maps any real-valued number into the range [0, 1].

### Forward Pass

```typescript
forward(inputs: number[]): number {
    if (inputs.length !== this.weights.length) {
        throw new Error("Input size does not match weight size");
    }

    const weightedSum = inputs.reduce((sum, input, i) => sum + input * this.weights[i], 0);
    return this.sigmoid(weightedSum + this.bias);
}
```

The forward pass:
1. Checks if the input size matches the number of weights.
2. Calculates the weighted sum of inputs.
3. Adds the bias to the weighted sum.
4. Applies the sigmoid activation function to the result.

This implementation provides the basic functionality of a neuron in a neural network, capable of processing inputs and producing an output.
