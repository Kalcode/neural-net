import { NeuralNetwork } from '../../src/NeuralNetwork';

export function runLargestExample() {
  // Largest training data
  const trainingData = [
    { input: [0, 0], target: [1] },
    { input: [0, 1], target: [1] },
    { input: [1, 0], target: [0] },
    { input: [1, 1], target: [1] },
    { input: [1, 2], target: [1] },
    { input: [2, 1], target: [0] },
    { input: [2, 2], target: [1] },
    { input: [2, 3], target: [1] },
    { input: [3, 2], target: [0] },
    { input: [3, 3], target: [1] },
    { input: [13, 4], target: [0] },
    { input: [4, 13], target: [1] },
    { input: [-13, 15], target: [1] },
    { input: [15, -13], target: [0] },
    { input: [100, 100], target: [1] },
    { input: [100, 101], target: [1] },
    { input: [101, 100], target: [0] },
    { input: [101, 101], target: [1] },
    { input: [1000, 1000], target: [1] },
    { input: [25, 1000], target: [1] },
    { input: [1000, 25], target: [0] },
    { input: [0, 1], target: [1] },
    { input: [2, 1], target: [0] },
    { input: [3, 5], target: [1] },
  ];

  // Create and train the neural network
  const nn = new NeuralNetwork(2, 3, 1, 0.01);
  const epochs = 10000;

  console.log("Training the neural network for LARGEST function...");
  for (let i = 0; i < epochs; i++) {
    trainingData.forEach(data => {
      const target = data.input[0] > data.input[1] ? [0] : [1];
      nn.train(data.input, target);
    });

    if (i % 1000 === 0) {
      const mse = trainingData.reduce((sum, data) => {
        const { output } = nn.forward(data.input);
        return sum + nn.meanSquaredError(output, data.target);
      }, 0) / trainingData.length;
      console.log(`Epoch ${i}: MSE = ${mse}`);
    }
  }

  console.log("\nTraining complete. Testing the trained network:");
  trainingData.forEach(data => {
    const { output } = nn.forward(data.input);
    console.log(`Input: [${data.input}], Output: ${output[0].toFixed(4)}, Target: ${data.target[0]}`);
  });

  console.log("\nTesting the trained network with random inputs:");
  for (let i = 0; i < 10; i++) {
    const a = Math.floor(Math.random() * 100);
    const b = Math.floor(Math.random() * 100);
    const { output } = nn.forward([a, b]);
    const predicted = Math.round(output[0]);
    const expected = a > b ? 0 : 1;
    console.log(`Input: [${a}, ${b}], Output: ${predicted}, Expected: ${expected}, Correct: ${predicted === expected}`);
  }

  // negative numbers
  console.log("\nTesting the trained network with negative inputs:");
  for (let i = 0; i < 10; i++) {
    const a = Math.floor(Math.random() * 100) - 100;
    const b = Math.floor(Math.random() * 100) - 100;
    const { output } = nn.forward([a, b]);
    const predicted = Math.round(output[0]);
    const expected = a > b ? 0 : 1;
    console.log(`Input: [${a}, ${b}], Output: ${predicted}, Expected: ${expected}, Correct: ${predicted === expected}`);
  }

  // negative mixed postivive
  console.log("\nTesting the trained network with mixed positive and negative inputs:");
  for (let i = 0; i < 10; i++) {
    const a = Math.floor(Math.random() * 100) - 50;
    const b = Math.floor(Math.random() * 100);
    const { output } = nn.forward([a, b]);
    const predicted = Math.round(output[0]);
    const expected = a > b ? 0 : 1;
    console.log(`Input: [${a}, ${b}], Output: ${predicted}, Expected: ${expected}, Correct: ${predicted === expected}`);
  }
}
