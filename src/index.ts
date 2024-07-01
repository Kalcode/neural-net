import { Network } from './Network';

console.log("\nXOR Network Example:");

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

// Create a XOR network
const inputSize = inputs[0].length;
const hiddenSizes = [inputSize, outputSize];
const outputSize = 1;
const xorNetwork = new Network(inputSize, hiddenSizes, outputSize);



// Train the network
console.log("Training the network...");
const epochs = 1000;
const learningRate = 0.01;
const momentum = 0.9;
const batchSize = inputs.length;
const maxGradientNorm = 1;

xorNetwork.train(inputs, targets, epochs, learningRate, momentum, batchSize, maxGradientNorm);

// Test the trained network
console.log("\nTesting the trained network:");
inputs.forEach(input => {
    const output = xorNetwork.forward(input);
    console.log(`${input[0]} XOR ${input[1]} = ${output[0].toFixed(4)}`);
});
