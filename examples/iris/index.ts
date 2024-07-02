import { NeuralNetwork } from '../../src/NeuralNetwork';
import { featuresToArray, classToArray, arrayToClass } from './iris_data';
import { loadIrisData } from './load_iris_data';
import type { IrisFeatures } from './iris_data';

export function runIrisExample() {
    const irisData = loadIrisData();
    
    // Create and train the neural network
    const nn = new NeuralNetwork(4, 5, 3);
    const epochs = 10000;

    console.log("Training the neural network for Iris classification...");
    for (let i = 0; i < epochs; i++) {
        irisData.forEach(data => {
            const input = featuresToArray(data.features);
            const target = classToArray(data.class);
            nn.train(input, target);
        });

        if (i % 1000 === 0) {
            const mse = irisData.reduce((sum, data) => {
                const input = featuresToArray(data.features);
                const target = classToArray(data.class);
                const { output } = nn.forward(input);
                return sum + nn.meanSquaredError(output, target);
            }, 0) / irisData.length;
            console.log(`Epoch ${i}: MSE = ${mse}`);
        }
    }

    console.log("\nTraining complete. Testing the trained network:");
    let correctPredictions = 0;
    irisData.forEach(data => {
        const { output } = nn.forward(featuresToArray(data.features));
        const predictedClass = arrayToClass(output);
        console.log(`Input: ${JSON.stringify(data.features)}`);
        console.log(`Predicted: ${predictedClass}, Actual: ${data.class}`);
        console.log(`Raw output: [${output.map(v => v.toFixed(4))}]\n`);
        
        if (predictedClass === data.class) {
            correctPredictions++;
        }
    });

    const accuracy = (correctPredictions / irisData.length) * 100;
    console.log(`\nAccuracy: ${accuracy.toFixed(2)}%`);

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
