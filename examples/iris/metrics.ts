import type { IrisClass } from './iris_data';

export interface ConfusionMatrix {
    [key: string]: { [key: string]: number };
}

export function createConfusionMatrix(actual: IrisClass[], predicted: IrisClass[]): ConfusionMatrix {
    const matrix: ConfusionMatrix = {
        'Iris-setosa': { 'Iris-setosa': 0, 'Iris-versicolor': 0, 'Iris-virginica': 0 },
        'Iris-versicolor': { 'Iris-setosa': 0, 'Iris-versicolor': 0, 'Iris-virginica': 0 },
        'Iris-virginica': { 'Iris-setosa': 0, 'Iris-versicolor': 0, 'Iris-virginica': 0 },
    };

    for (let i = 0; i < actual.length; i++) {
        matrix[actual[i]][predicted[i]]++;
    }

    return matrix;
}

export function calculateMetrics(confusionMatrix: ConfusionMatrix) {
    const classes = Object.keys(confusionMatrix);
    const metrics: { [key: string]: { precision: number; recall: number; f1Score: number } } = {};

    for (const cls of classes) {
        let truePositives = confusionMatrix[cls][cls];
        let falsePositives = 0;
        let falseNegatives = 0;

        for (const otherCls of classes) {
            if (otherCls !== cls) {
                falsePositives += confusionMatrix[otherCls][cls];
                falseNegatives += confusionMatrix[cls][otherCls];
            }
        }

        const precision = truePositives / (truePositives + falsePositives);
        const recall = truePositives / (truePositives + falseNegatives);
        const f1Score = 2 * (precision * recall) / (precision + recall);

        metrics[cls] = { precision, recall, f1Score };
    }

    return metrics;
}

export function printConfusionMatrix(matrix: ConfusionMatrix) {
    const classes = Object.keys(matrix);
    console.log('Confusion Matrix:');
    console.log('                 Predicted');
    console.log('Actual         ' + classes.join('   '));
    for (const actualClass of classes) {
        let row = `${actualClass.padEnd(15)}`;
        for (const predictedClass of classes) {
            row += `${matrix[actualClass][predictedClass].toString().padStart(3)}   `;
        }
        console.log(row);
    }
}

export function printMetrics(metrics: ReturnType<typeof calculateMetrics>) {
    console.log('\nMetrics:');
    for (const [cls, { precision, recall, f1Score }] of Object.entries(metrics)) {
        console.log(`${cls}:`);
        console.log(`  Precision: ${precision.toFixed(4)}`);
        console.log(`  Recall:    ${recall.toFixed(4)}`);
        console.log(`  F1-Score:  ${f1Score.toFixed(4)}`);
    }
}
