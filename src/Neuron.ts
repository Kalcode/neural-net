import { isValidNumber } from './utils';

export class Neuron {
    public weights: number[];
    public bias: number;

    constructor(inputSize: number) {
        this.weights = Array.from({ length: inputSize }, () => Math.random() * 2 - 1);
        this.bias = Math.random() * 2 - 1;
    }

    private sigmoid(x: number): number {
        if (x < -709) return 0;
        if (x > 709) return 1;
        const result = 1 / (1 + Math.exp(-x));
        if (!isValidNumber(result)) {
            console.error(`Invalid sigmoid result: ${result} for input ${x}`);
            return 0.5; // Default to middle of sigmoid range
        }
        return result;
    }

    forward(inputs: number[]): number {
        if (inputs.length !== this.weights.length) {
            throw new Error("Input size does not match weight size");
        }

        const weightedSum = inputs.reduce((sum, input, i) => {
            const product = input * this.weights[i];
            if (!isValidNumber(product)) {
                console.error(`Invalid product: ${input} * ${this.weights[i]} = ${product}`);
                return sum;
            }
            return sum + product;
        }, 0) + this.bias;

        if (!isValidNumber(weightedSum)) {
            console.error(`Invalid weighted sum: ${weightedSum}`);
            return 0.5; // Default to middle of output range
        }

        return this.sigmoid(weightedSum);
    }

    updateWeights(error: number, learningRate: number, inputs: number[]): void {
        const output = this.forward(inputs);
        const delta = error * output * (1 - output);

        if (!isValidNumber(delta)) {
            console.error(`Invalid delta: ${delta}, error: ${error}, output: ${output}`);
            return;
        }

        for (let i = 0; i < this.weights.length; i++) {
            const weightDelta = learningRate * delta * inputs[i];
            if (!isValidNumber(weightDelta)) {
                console.error(`Invalid weight delta: ${weightDelta}`);
                continue;
            }
            this.weights[i] += weightDelta;
        }
        
        const biasDelta = learningRate * delta;
        if (isValidNumber(biasDelta)) {
            this.bias += biasDelta;
        } else {
            console.error(`Invalid bias delta: ${biasDelta}`);
        }
    }
}
