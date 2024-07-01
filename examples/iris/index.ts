import { NeuralNetwork } from '../../src/NeuralNetwork';

// Iris dataset (simplified version with 3 features)
const irisData = [
    { input: [5.1, 3.5, 1.4], target: [1, 0, 0] }, // Setosa
    { input: [4.9, 3.0, 1.4], target: [1, 0, 0] },
    { input: [7.0, 3.2, 4.7], target: [0, 1, 0] }, // Versicolor
    { input: [6.4, 3.2, 4.5], target: [0, 1, 0] },
    { input: [6.3, 3.3, 6.0], target: [0, 0, 1] }, // Virginica
    { input: [5.8, 2.7, 5.1], target: [0, 0, 1] },
];

export function runIrisExample() {
    // Create and train the neural network
    const nn = new NeuralNetwork(3, 5, 3);
    const epochs = 10000;

    console.log("Training the neural network for Iris classification...");
    for (let i = 0; i < epochs; i++) {
        irisData.forEach(data => {
            nn.train(data.input, data.target);
        });

        if (i % 1000 === 0) {
            const mse = irisData.reduce((sum, data) => {
                const { output } = nn.forward(data.input);
                return sum + nn.meanSquaredError(output, data.target);
            }, 0) / irisData.length;
            console.log(`Epoch ${i}: MSE = ${mse}`);
        }
    }

    console.log("\nTraining complete. Testing the trained network:");
    irisData.forEach(data => {
        const { output } = nn.forward(data.input);
        console.log(`Input: [${data.input}], Output: [${output.map(v => v.toFixed(4))}], Target: [${data.target}]`);
    });
}
