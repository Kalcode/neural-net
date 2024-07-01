import { Neuron } from './Neuron';
import BigNumber from 'bignumber.js';
import { toBigNumber } from './utils';

export class Layer {
    public neurons: Neuron[];
    private prevWeightDeltas: BigNumber[][];
    private prevBiasDeltas: BigNumber[];

    constructor(inputSize: number, outputSize: number) {
        this.neurons = Array.from({ length: outputSize }, () => new Neuron(inputSize));
        this.prevWeightDeltas = Array.from({ length: outputSize }, () => Array(inputSize).fill(toBigNumber(0)));
        this.prevBiasDeltas = Array(outputSize).fill(toBigNumber(0));
    }

    forward(inputs: BigNumber[]): BigNumber[] {
        return this.neurons.map(neuron => neuron.forward(inputs));
    }

    train(errors: BigNumber[], learningRate: BigNumber, inputs: BigNumber[], momentum: BigNumber, batchSize: BigNumber, maxGradientNorm: BigNumber): void {
        console.log(`Layer training:`);
        console.log(`Errors:`, errors.map(e => e.toString()));
        console.log(`Learning rate:`, learningRate.toString());
        console.log(`Inputs:`, inputs.map(i => i.toString()));
        console.log(`Batch size:`, batchSize.toString());
        console.log(`Momentum:`, momentum.toString());
        console.log(`Max gradient norm:`, maxGradientNorm.toString());

        this.neurons.forEach((neuron, i) => {
            console.log(`Training neuron ${i}:`);
            const { weightDeltas, biasDelta } = neuron.calculateGradients(errors[i], inputs);
            
            // Apply momentum
            const newWeightDeltas = weightDeltas.map((delta, j) => delta.plus(momentum.times(this.prevWeightDeltas[i][j])));
            const newBiasDelta = biasDelta.plus(momentum.times(this.prevBiasDeltas[i]));

            // Clip gradients
            const gradients = [...newWeightDeltas, newBiasDelta];
            const gradientNorm = toBigNumber(Math.sqrt(gradients.reduce((sum, g) => sum + g.pow(2).toNumber(), 0)));
            const scalingFactor = BigNumber.minimum(toBigNumber(1), maxGradientNorm.dividedBy(gradientNorm));

            // Apply clipped gradients
            neuron.applyGradients(newWeightDeltas.map(d => d.times(scalingFactor)), newBiasDelta.times(scalingFactor), learningRate);

            // Store current deltas for next iteration
            this.prevWeightDeltas[i] = newWeightDeltas;
            this.prevBiasDeltas[i] = newBiasDelta;
        });
    }
}
