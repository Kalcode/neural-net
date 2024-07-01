import { round } from './utils';

export class Neuron {
    private weights: number[];
    private bias: number;

    constructor(inputSize: number) {
        // Initialize weights randomly between -1 and 1
        this.weights = Array.from({ length: inputSize }, () => round(Math.random() * 2 - 1));
        // Initialize bias randomly between -1 and 1
        this.bias = round(Math.random() * 2 - 1);
    }

    // Activation function (using sigmoid for this example)
    private sigmoid(x: number): number {
        return round(1 / (1 + Math.exp(-x)));
    }

    // Forward pass
    forward(inputs: number[]): number {
        if (inputs.length !== this.weights.length) {
            throw new Error("Input size does not match weight size");
        }

        // Calculate the weighted sum of inputs
        const weightedSum = round(inputs.reduce((sum, input, i) => sum + input * this.weights[i], 0));
        
        // Add bias and apply activation function
        return this.sigmoid(round(weightedSum + this.bias));
    }
}
