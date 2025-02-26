import * as fs from 'fs';
import * as path from 'path';

import { NeuralNetwork } from '../../src/NeuralNetwork';

import { createConfusionMatrix, calculateMetrics, printConfusionMatrix, printMetrics } from './metrics';
import { featuresToArray, classToArray, arrayToClass } from './iris_data';
import { generateHTML } from './html_generator';
import { loadIrisData } from './load_iris_data';
import type { IrisFeatures, IrisClass } from './iris_data';


export function runIrisExample() {
    const irisData = loadIrisData();

    // Create and train the neural network
    const nn = new NeuralNetwork(4, 8, 3, 0.01); // Increased hidden nodes, lower learning rate
    const epochs = 20000; // Increased number of epochs
    const learningCurve: number[] = [];

    console.log("Training the neural network for Iris classification...");
    for (let i = 0; i < epochs; i++) {
        let epochError = 0;
        irisData.forEach(data => {
            const input = featuresToArray(data.features);
            const target = classToArray(data.class);
            if (input.length === 4 && target.length === 3) {
                const error = nn.train(input, target);
                epochError += error;
            } else {
                console.error(`Invalid input or target length for data point:`, data);
            }
        });

        const mse = epochError / irisData.length;
        learningCurve.push(mse);

        if (i % 1000 === 0) {
            console.log(`Epoch ${i}: MSE = ${mse.toFixed(6)}`);
        }
    }

    console.log("\nTraining complete. Testing the trained network:");
    const actualClasses: IrisClass[] = [];
    const predictedClasses: IrisClass[] = [];
    irisData.forEach(data => {
        const { output } = nn.forward(featuresToArray(data.features));
        const predictedClass = arrayToClass(output);
        actualClasses.push(data.class);
        predictedClasses.push(predictedClass);
    });

    const confusionMatrix = createConfusionMatrix(actualClasses, predictedClasses);
    const metrics = calculateMetrics(confusionMatrix);

    printConfusionMatrix(confusionMatrix);
    printMetrics(metrics);

    const accuracy = predictedClasses.filter((pred, i) => pred === actualClasses[i]).length / irisData.length * 100;
    console.log(`\nOverall Accuracy: ${accuracy.toFixed(2)}%`);

    // Print learning curve
    console.log("\nLearning Curve (MSE every 1000 epochs):");
    learningCurve.filter((_, i) => i % 1000 === 0).forEach((mse, i) => {
        console.log(`Epoch ${i * 1000}: ${mse.toFixed(6)}`);
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

    // Generate HTML report
    const htmlContent = generateHTML(confusionMatrix, metrics, learningCurve, accuracy);

    const htmlFilePath = path.join(__dirname, 'data', 'iris_classification_results.html');
    fs.writeFileSync(htmlFilePath, htmlContent);
    console.log(`\nHTML report generated: ${htmlFilePath}`);

    // open file
    const { exec } = require('child_process');
    switch (process.platform) {
        case 'darwin':
            exec(`open ${htmlFilePath}`);
            break;
        default:
            console.log(`Please open the HTML report manually: ${htmlFilePath}`);
            break;
    }

    nn.print();

}
