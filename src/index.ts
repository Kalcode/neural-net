import { Network } from './Network';
import { toBigNumber } from './utils';
import BigNumber from 'bignumber.js';

console.log("\nXOR Network Example:");

// Create a XOR network
const xorNetwork = new Network([2, 2, 1]);

// XOR training data
const inputs = [
    [toBigNumber(0), toBigNumber(0)],
    [toBigNumber(0), toBigNumber(1)],
    [toBigNumber(1), toBigNumber(0)],
    [toBigNumber(1), toBigNumber(1)]
];
const targets = [
    [toBigNumber(0)],
    [toBigNumber(1)],
    [toBigNumber(1)],
    [toBigNumber(0)]
];

// Train the network
console.log("Training the network...");
const epochs = 1000;
const learningRate = toBigNumber(0.1);
const momentum = toBigNumber(0.9);
const batchSize = toBigNumber(1);
const maxGradientNorm = toBigNumber(1);

xorNetwork.train(inputs, targets, epochs, learningRate, momentum, batchSize, maxGradientNorm);

// Test the trained network
console.log("\nTesting the trained network:");
inputs.forEach(input => {
    const output = xorNetwork.forward(input);
    console.log(`${input[0].toString()} XOR ${input[1].toString()} = ${output[0].toFixed(4)}`);
});
