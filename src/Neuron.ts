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
        const result = toBigNumber(1).dividedBy(toBigNumber(1).plus(x.negated().exp()));
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

    updateWeights(error: BigNumber, learningRate: BigNumber, inputs: BigNumber[], batchSize: BigNumber): void {
        console.log(`Neuron updateWeights:`);
        console.log(`Error:`, error.toString());
        console.log(`Learning rate:`, learningRate.toString());
        console.log(`Inputs:`, inputs.map(i => i.toString()));
        
        const output = this.forward(inputs);
        console.log(`Output:`, output.toString());
        
        // Add a small constant to prevent division by zero
        const delta = error.times(output).times(toBigNumber(1).minus(output).plus(toBigNumber(1e-7)));
        console.log(`Delta:`, delta.toString());

        if (!isValidNumber(delta)) {
            console.error(`Invalid delta: ${delta.toString()}, error: ${error.toString()}, output: ${output.toString()}`);
            return;
        }

        for (let i = 0; i < this.weights.length; i++) {
            const weightDelta = learningRate.times(delta).times(inputs[i]);
            console.log(`Weight ${i} delta:`, weightDelta.toString());
            if (!isValidNumber(weightDelta)) {
                console.error(`Invalid weight delta: ${weightDelta.toString()}`);
                continue;
            }
            this.weights[i] = this.weights[i].plus(weightDelta);
        }
        
        const biasDelta = learningRate.times(delta);
        console.log(`Bias delta:`, biasDelta.toString());
        if (isValidNumber(biasDelta)) {
            this.bias = this.bias.plus(biasDelta);
        } else {
            console.error(`Invalid bias delta: ${biasDelta.toString()}`);
        }

        this.updateBatchNormParams(error, learningRate, inputs, batchSize);
    }

    private updateBatchNormParams(error: BigNumber, learningRate: BigNumber, inputs: BigNumber[], batchSize: BigNumber): void {
        const meanDelta = error.times(this.gamma).dividedBy(batchSize.sqrt());
        this.runningMean = this.runningMean.minus(learningRate.times(meanDelta));

        const varianceDelta = error.times(this.gamma).dividedBy(batchSize.sqrt().times(this.runningVar.sqrt()));
        this.runningVar = this.runningVar.minus(learningRate.times(varianceDelta));

        const gammaDelta = error.times(this.batchNormalize(inputs[0])).dividedBy(batchSize);
        this.gamma = this.gamma.minus(learningRate.times(gammaDelta));

        const betaDelta = error.dividedBy(batchSize);
        this.beta = this.beta.minus(learningRate.times(betaDelta));
    }

    // Add a method to clip weights and bias to prevent exploding gradients
    clipWeightsAndBias(min: BigNumber = toBigNumber(-1), max: BigNumber = toBigNumber(1)): void {
        this.weights = this.weights.map(w => BigNumber.maximum(min, BigNumber.minimum(max, w)));
        this.bias = BigNumber.maximum(min, BigNumber.minimum(max, this.bias));
    }
}
