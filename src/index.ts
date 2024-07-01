import { Network } from './Network';

console.log("\nXOR Network Example:");

// Create a XOR network
const inputSize = inputs[0].length;
const hiddenSizes = [2];  
const outputSize = 1;
const xorNetwork = new Network(inputSize, hiddenSizes, outputSize);

// XOR training data
const inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
];
const targets = [
    [0],
    [1],
    [1],
    [0]
];

// Train the network
console.log("Training the network...");
const epochs = 1000;
const learningRate = 0.1;
const momentum = 0.9;
const batchSize = 1;
const maxGradientNorm = 1;

xorNetwork.train(inputs, targets, epochs, learningRate, momentum, batchSize, maxGradientNorm);

// Test the trained network
console.log("\nTesting the trained network:");
inputs.forEach(input => {
    const output = xorNetwork.forward(input);
    console.log(`${input[0]} XOR ${input[1]} = ${output[0].toFixed(4)}`);
});
