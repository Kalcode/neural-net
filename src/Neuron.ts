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
        if (x < -709) return 0;
        if (x > 709) return 1;
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

    updateWeights(error: number, learningRate: number): number[] {
        if (!isValidNumber(error) || !isValidNumber(learningRate)) {
            throw new Error(`Invalid error or learning rate: error=${error}, learningRate=${learningRate}`);
        }

        const delta = this.clipGradient(error * this.sigmoidDerivative(this.lastOutput));
        if (!isValidNumber(delta)) {
            throw new Error(`Invalid delta: ${delta}, error: ${error}, lastOutput: ${this.lastOutput}`);
        }

        const deltas = this.weights.map((weight, i) => {
            if (!isValidNumber(this.lastInputs[i])) {
                throw new Error(`Invalid last input at index ${i}: ${this.lastInputs[i]}`);
            }
            const update = round(learningRate * delta * this.lastInputs[i]);
            if (!isValidNumber(update)) {
                console.warn(`Warning: Invalid weight update. Using 0 instead. Debug info: learningRate=${learningRate}, delta=${delta}, lastInput=${this.lastInputs[i]}`);
                return 0;
            }
            this.weights[i] = round(weight + update);
            return delta * weight;
        });

        const biasUpdate = round(learningRate * delta);
        if (!isValidNumber(biasUpdate)) {
            console.warn(`Warning: Invalid bias update. Using 0 instead. Debug info: learningRate=${learningRate}, delta=${delta}`);
        } else {
            this.bias = round(this.bias + biasUpdate);
        }

        return deltas;
    }
}
