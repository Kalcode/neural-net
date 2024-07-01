import { round, isValidNumber } from './utils';

export class Neuron {
    public weights: number[];
    public bias: number;
    private lastInputs: number[];
    private lastOutput: number;

    constructor(inputSize: number) {
        this.weights = Array.from({ length: inputSize }, () => round(Math.random() * 2 - 1));
        this.bias = round(Math.random() * 2 - 1);
        this.lastInputs = [];
        this.lastOutput = 0;
    }

    private sigmoid(x: number): number {
        // Clip x to prevent extreme values
        x = Math.max(-709, Math.min(709, x));
        return round(1 / (1 + Math.exp(-x)));
    }

    private sigmoidDerivative(x: number): number {
        return round(x * (1 - x));
    }

    private clipGradient(gradient: number, clipValue: number = 1): number {
        return Math.max(-clipValue, Math.min(clipValue, gradient));
    }

    forward(inputs: number[]): number {
        if (inputs.length !== this.weights.length) {
            throw new Error("Input size does not match weight size");
        }

        this.lastInputs = inputs;
        const weightedSum = round(inputs.reduce((sum, input, i) => {
            if (!isValidNumber(input) || !isValidNumber(this.weights[i])) {
                throw new Error(`Invalid input or weight: input=${input}, weight=${this.weights[i]}`);
            }
            return sum + input * this.weights[i];
        }, 0) + this.bias);

        this.lastOutput = this.sigmoid(weightedSum);
        return this.lastOutput;
    }

    updateWeights(error: number, learningRate: number): void {
        if (!isValidNumber(error) || !isValidNumber(learningRate)) {
            throw new Error(`Invalid error or learning rate: error=${error}, learningRate=${learningRate}`);
        }

        const delta = this.clipGradient(error * this.sigmoidDerivative(this.lastOutput));
        this.weights = this.weights.map((weight, i) => {
            const update = round(learningRate * delta * this.lastInputs[i]);
            if (!isValidNumber(update)) {
                throw new Error(`Invalid weight update: update=${update}, weight=${weight}, input=${this.lastInputs[i]}`);
            }
            return round(weight + update);
        });
        this.bias = round(this.bias + learningRate * delta);
    }
}
