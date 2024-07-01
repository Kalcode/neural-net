import { NeuralNetwork } from '../../src/NeuralNetwork';
import { irisData, featuresToArray, classToArray, arrayToClass, IrisFeatures, IrisClass } from './iris_data';

export function runIrisExample() {
    // Create and train the neural network
    const nn = new NeuralNetwork(4, 5, 3);
    const epochs = 10000;

    console.log("Training the neural network for Iris classification...");
    for (let i = 0; i < epochs; i++) {
        irisData.forEach(data => {
            nn.train(featuresToArray(data.features), classToArray(data.class));
        });

        if (i % 1000 === 0) {
            const mse = irisData.reduce((sum, data) => {
                const { output } = nn.forward(featuresToArray(data.features));
                return sum + nn.meanSquaredError(output, classToArray(data.class));
            }, 0) / irisData.length;
            console.log(`Epoch ${i}: MSE = ${mse}`);
        }
    }

    console.log("\nTraining complete. Testing the trained network:");
    irisData.forEach(data => {
        const { output } = nn.forward(featuresToArray(data.features));
        const predictedClass = arrayToClass(output);
        console.log(`Input: ${JSON.stringify(data.features)}`);
        console.log(`Predicted: ${predictedClass}, Actual: ${data.class}`);
        console.log(`Raw output: [${output.map(v => v.toFixed(4))}]\n`);
    });

    // Example of using the trained network with user input
    console.log("\nTry classifying a new Iris flower:");
    const newIris: IrisFeatures = {
        sepalLength: 5.5,
        sepalWidth: 2.6,
        petalLength: 4.4,
        petalWidth: 1.2
    };
    const { output } = nn.forward(featuresToArray(newIris));
    const predictedClass = arrayToClass(output);
    console.log(`Input: ${JSON.stringify(newIris)}`);
    console.log(`Predicted class: ${predictedClass}`);
    console.log(`Raw output: [${output.map(v => v.toFixed(4))}]`);
}
