import { NeuralNetwork } from '../../src/NeuralNetwork';

export function runANDExample() {
    // AND training data
    const trainingData = [
        { input: [0, 0], target: [0] },
        { input: [0, 1], target: [0] },
        { input: [1, 0], target: [0] },
        { input: [1, 1], target: [1] }
    ];

    // Create and train the neural network
    const nn = new NeuralNetwork(2, 3, 1);
    const epochs = 10000;

    console.log("Training the neural network for AND function...");
    for (let i = 0; i < epochs; i++) {
        trainingData.forEach(data => {
            nn.train(data.input, data.target);
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
}
