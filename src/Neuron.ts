import { isValidNumber, toBigNumber } from './utils';
import BigNumber from 'bignumber.js';

export class Neuron {
    public weights: BigNumber[];
    public bias: BigNumber;
    private gamma: BigNumber;
    private beta: BigNumber;
    private runningMean: BigNumber;
    private runningVar: BigNumber;

    constructor(inputSize: number) {
        this.weights = Array.from({ length: inputSize }, () => toBigNumber(Math.random() * 2 - 1));
        this.bias = toBigNumber(Math.random() * 2 - 1);
        this.gamma = toBigNumber(1);
        this.beta = toBigNumber(0);
        this.runningMean = toBigNumber(0);
        this.runningVar = toBigNumber(1);
    }

    private sigmoid(x: BigNumber): BigNumber {
        if (x.isLessThan(-709)) return toBigNumber(0);
        if (x.isGreaterThan(709)) return toBigNumber(1);
        const exp = Math.exp(x.negated().toNumber());
        const result = toBigNumber(1).dividedBy(toBigNumber(1).plus(toBigNumber(exp)));
        if (!isValidNumber(result)) {
            console.error(`Invalid sigmoid result: ${result.toString()} for input ${x.toString()}`);
            return toBigNumber(0.5); // Default to middle of sigmoid range
        }
        return result;
    }

    private batchNormalize(x: BigNumber): BigNumber {
        const normalized = x.minus(this.runningMean).dividedBy(this.runningVar.sqrt());
        return normalized.times(this.gamma).plus(this.beta);
    }

    forward(inputs: BigNumber[]): BigNumber {
        if (inputs.length !== this.weights.length) {
            throw new Error("Input size does not match weight size");
        }

        const weightedSum = inputs.reduce((sum, input, i) => {
            const product = input.times(this.weights[i]);
            if (!isValidNumber(product)) {
                console.error(`Invalid product: ${input.toString()} * ${this.weights[i].toString()} = ${product.toString()}`);
                return sum;
            }
            return sum.plus(product);
        }, toBigNumber(0)).plus(this.bias);

        if (!isValidNumber(weightedSum)) {
            console.error(`Invalid weighted sum: ${weightedSum.toString()}`);
            return toBigNumber(0.5); // Default to middle of output range
        }

        const normalized = this.batchNormalize(weightedSum);
        return this.sigmoid(normalized);
    }

    calculateGradients(error: BigNumber, inputs: BigNumber[]): { weightDeltas: BigNumber[], biasDelta: BigNumber } {
        if (!isValidNumber(error)) {
            console.error(`Invalid error in calculateGradients: ${error}`);
            return { weightDeltas: this.weights.map(() => toBigNumber(0)), biasDelta: toBigNumber(0) };
        }

        const output = this.forward(inputs);
        
        // Add a small constant to prevent division by zero
        const delta = error.times(output).times(toBigNumber(1).minus(output).plus(toBigNumber(1e-7)));

        if (!isValidNumber(delta)) {
            console.error(`Invalid delta: ${delta.toString()}, error: ${error.toString()}, output: ${output.toString()}`);
            return { weightDeltas: this.weights.map(() => toBigNumber(0)), biasDelta: toBigNumber(0) };
        }

        const weightDeltas = this.weights.map((_, i) => delta.times(inputs[i]));
        const biasDelta = delta;

        return { weightDeltas, biasDelta };
    }

    applyGradients(weightDeltas: BigNumber[], biasDelta: BigNumber, learningRate: BigNumber): void {
        this.weights = this.weights.map((w, i) => w.plus(weightDeltas[i].times(learningRate)));
        this.bias = this.bias.plus(biasDelta.times(learningRate));

        this.updateBatchNormParams(biasDelta, learningRate);
    }

    private updateBatchNormParams(error: BigNumber, learningRate: BigNumber): void {
        const meanDelta = error.times(this.gamma);
        this.runningMean = this.runningMean.minus(learningRate.times(meanDelta));

        const varianceDelta = error.times(this.gamma).dividedBy(this.runningVar.sqrt());
        this.runningVar = this.runningVar.minus(learningRate.times(varianceDelta));

        const gammaDelta = error.times(this.batchNormalize(this.bias));
        this.gamma = this.gamma.minus(learningRate.times(gammaDelta));

        const betaDelta = error;
        this.beta = this.beta.minus(learningRate.times(betaDelta));
    }

    // Add a method to clip weights and bias to prevent exploding gradients
    clipWeightsAndBias(min: BigNumber = toBigNumber(-1), max: BigNumber = toBigNumber(1)): void {
        this.weights = this.weights.map(w => BigNumber.maximum(min, BigNumber.minimum(max, w)));
        this.bias = BigNumber.maximum(min, BigNumber.minimum(max, this.bias));
    }
}
