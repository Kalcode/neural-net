import { Neuron } from './Neuron';
import BigNumber from 'bignumber.js';
import { toBigNumber } from './utils';

export class Layer {
    public neurons: Neuron[];

    constructor(inputSize: number, outputSize: number) {
        this.neurons = Array.from({ length: outputSize }, () => new Neuron(inputSize));
    }

    forward(inputs: BigNumber[]): BigNumber[] {
        return this.neurons.map(neuron => neuron.forward(inputs));
    }

    train(errors: BigNumber[], learningRate: BigNumber, inputs: BigNumber[], batchSize: BigNumber): void {
        console.log(`Layer training:`);
        console.log(`Errors:`, errors.map(e => e.toString()));
        console.log(`Learning rate:`, learningRate.toString());
        console.log(`Inputs:`, inputs.map(i => i.toString()));
        console.log(`Batch size:`, batchSize.toString());
        this.neurons.forEach((neuron, i) => {
            console.log(`Training neuron ${i}:`);
            neuron.updateWeights(errors[i], learningRate, inputs, batchSize);
            neuron.clipWeightsAndBias(toBigNumber(-1), toBigNumber(1));
        });
    }
}
