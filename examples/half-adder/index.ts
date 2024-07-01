import { NeuralNetwork } from '../../src/NeuralNetwork';

export function runHalfAdderExample() {
    // Half Adder training data
    const trainingData = [
        { input: [0, 0], target: [0, 0] }, // Sum: 0, Carry: 0
        { input: [0, 1], target: [1, 0] }, // Sum: 1, Carry: 0
        { input: [1, 0], target: [1, 0] }, // Sum: 1, Carry: 0
        { input: [1, 1], target: [0, 1] }  // Sum: 0, Carry: 1
    ];

    // Create and train the neural network
    const nn = new NeuralNetwork(2, 4, 2);
    const epochs = 20000;

    console.log("Training the neural network for Half Adder function...");
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
        console.log(`Input: [${data.input}], Output: [${output.map(v => v.toFixed(4))}], Target: [${data.target}]`);
    });
}
