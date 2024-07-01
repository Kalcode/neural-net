import { isValidNumber } from './utils';

export class Neuron {
    public weights: number[];
    public bias: number;
    private gamma: number;
    private beta: number;
    private runningMean: number;
    private runningVar: number;

    constructor(inputSize: number) {
        this.weights = Array.from({ length: inputSize }, () => Math.random() * 2 - 1);
        this.bias = Math.random() * 2 - 1;
        this.gamma = 1;
        this.beta = 0;
        this.runningMean = 0;
        this.runningVar = 1;
    }

    private sigmoid(x: number): number {
        return 1 / (1 + Math.exp(-x));
    }

    private batchNormalize(x: number): number {
        const normalized = (x - this.runningMean) / Math.sqrt(this.runningVar);
        return normalized * this.gamma + this.beta;
    }

    forward(inputs: number[]): number {
        if (inputs.length !== this.weights.length) {
            throw new Error(`Input size (${inputs.length}) does not match weight size (${this.weights.length}).`);
        }

        const weightedSum = inputs.reduce((sum, input, i) => {
            if (!isValidNumber(input) || !isValidNumber(this.weights[i])) {
                console.warn(`Invalid input or weight: input=${input}, weight=${this.weights[i]}`);
                return sum;
            }
            return sum + input * this.weights[i];
        }, 0) + this.bias;

        const normalized = this.batchNormalize(weightedSum);
        return this.sigmoid(normalized);
    }

    calculateGradients(error: number, inputs: number[]): { weightDeltas: number[], biasDelta: number } {
        if (!isValidNumber(error)) {
            console.error(`Invalid error in calculateGradients: ${error}`);
            return { weightDeltas: this.weights.map(() => 0), biasDelta: 0 };
        }

        try {
            const output = this.forward(inputs);

            if (!isValidNumber(output)) {
                console.error(`Invalid output in calculateGradients: ${output}`);
                return { weightDeltas: this.weights.map(() => 0), biasDelta: 0 };
            }
            
            // Add a small constant to prevent division by zero
            const delta = error * output * (1 - output + 1e-7);

            if (!isValidNumber(delta)) {
                console.error(`Invalid delta: ${delta}, error: ${error}, output: ${output}`);
                return { weightDeltas: this.weights.map(() => 0), biasDelta: 0 };
            }

            const weightDeltas = this.weights.map((_, i) => {
                const wd = delta * inputs[i];
                return isValidNumber(wd) ? wd : 0;
            });
            const biasDelta = delta;

            return { weightDeltas, biasDelta };
        } catch (error) {
            console.error(`Error in calculateGradients: ${error}`);
            return { weightDeltas: this.weights.map(() => 0), biasDelta: 0 };
        }
    }

    applyGradients(weightDeltas: number[], biasDelta: number, learningRate: number): void {
        this.weights = this.weights.map((w, i) => w + weightDeltas[i] * learningRate);
        this.bias += biasDelta * learningRate;

        this.updateBatchNormParams(biasDelta, learningRate);
    }

    private updateBatchNormParams(error: number, learningRate: number): void {
        const epsilon = 1e-8; // Small constant to avoid division by zero

        const meanDelta = error * this.gamma;
        this.runningMean -= learningRate * meanDelta;

        const varianceDelta = error * this.gamma / Math.sqrt(this.runningVar + epsilon);
        this.runningVar -= learningRate * varianceDelta;

        const gammaDelta = error * this.batchNormalize(this.bias);
        this.gamma -= learningRate * gammaDelta;

        const betaDelta = error;
        this.beta -= learningRate * betaDelta;
    }

    // Add a method to clip weights and bias to prevent exploding gradients
    clipWeightsAndBias(min: number = -1, max: number = 1): void {
        this.weights = this.weights.map(w => Math.max(min, Math.min(max, w)));
        this.bias = Math.max(min, Math.min(max, this.bias));
    }
}
