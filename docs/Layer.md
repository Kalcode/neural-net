# Layer Implementation

The `Layer` class represents a layer of neurons in our neural network. It's implemented in the `src/Layer.ts` file.

## Class Structure

```typescript
import { Neuron } from './Neuron';

export class Layer {
    private neurons: Neuron[];

    constructor(inputSize: number, outputSize: number) {
        // ...
    }

    forward(inputs: number[]): number[] {
        // ...
    }
}
```

## Key Components

1. **Neurons**: A layer contains an array of `Neuron` instances.

2. **Forward Pass**: The `forward` method processes inputs through all neurons in the layer.

## Implementation Details

### Constructor

```typescript
constructor(inputSize: number, outputSize: number) {
    this.neurons = Array.from({ length: outputSize }, () => new Neuron(inputSize));
}
```

The constructor creates an array of `Neuron` instances. The number of neurons is determined by the `outputSize` parameter, and each neuron is initialized with `inputSize` inputs.

### Forward Pass

```typescript
forward(inputs: number[]): number[] {
    return this.neurons.map(neuron => neuron.forward(inputs));
}
```

The forward pass:
1. Takes an array of inputs.
2. Passes these inputs through each neuron in the layer.
3. Returns an array of outputs, one from each neuron.

This implementation allows the layer to process a set of inputs through all of its neurons in parallel, producing a set of outputs that can be used as inputs for the next layer in the network.

## Usage in the Network

Layers are used to group neurons and simplify the structure of the neural network. By using layers, we can easily create networks with multiple layers of different sizes, allowing for more complex computations and representations.

In a full neural network implementation, you would typically have:
1. An input layer
2. One or more hidden layers
3. An output layer

Each layer would process inputs from the previous layer and pass its outputs to the next layer, allowing the network to learn and represent complex patterns in the data.
