import { Layer } from './Layer';

// Create a simple XOR network
// Input layer: 2 neurons (for 2 inputs)
// Hidden layer: 2 neurons
// Output layer: 1 neuron

const inputLayer = new Layer(2, 2);
const outputLayer = new Layer(2, 1);

function xorNetwork(input1: number, input2: number): number {
    const hiddenOutputs = inputLayer.forward([input1, input2]);
    const finalOutput = outputLayer.forward(hiddenOutputs);
    return finalOutput[0]; // We only have one output neuron
}

// Test the XOR network
console.log("XOR Network Test:");
console.log(`0 XOR 0 = ${xorNetwork(0, 0)}`);
console.log(`0 XOR 1 = ${xorNetwork(0, 1)}`);
console.log(`1 XOR 0 = ${xorNetwork(1, 0)}`);
console.log(`1 XOR 1 = ${xorNetwork(1, 1)}`);

// Note: This network is not trained, so the outputs will be random.
// In a fully implemented neural network, we would need to train the
// network on XOR data to get accurate results.
