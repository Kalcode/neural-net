import { Network } from './Network';

console.log("\nXOR Network Example:");

// Create a XOR network
const xorNetwork = new Network([2, 2, 1]);

// XOR training data
const inputs = [[0, 0], [0, 1], [1, 0], [1, 1]];
const targets = [[0], [1], [1], [0]];

// Train the network
console.log("Training the network...");
xorNetwork.train(inputs, targets, 10000, 0.1);

// Test the trained network
console.log("\nTesting the trained network:");
inputs.forEach(input => {
    const output = xorNetwork.forward(input);
    console.log(`${input[0]} XOR ${input[1]} = ${output[0].toFixed(4)}`);
});
