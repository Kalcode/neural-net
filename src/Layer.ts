import { Neuron } from './Neuron';
import { isValidNumber } from './utils';

export class Layer {
    public neurons: Neuron[];

    constructor(inputSize: number, outputSize: number) {
        this.neurons = Array.from({ length: outputSize }, () => new Neuron(inputSize));
    }

    forward(inputs: number[]): number[] {
        return this.neurons.map(neuron => {
            const output = neuron.forward(inputs);
            if (!isValidNumber(output)) {
                throw new Error(`Invalid neuron output: ${output}`);
            }
            return output;
        });
    }

    backpropagate(errors: number[], learningRate: number): number[] {
        return this.neurons.map((neuron, i) => {
            if (!isValidNumber(errors[i])) {
                throw new Error(`Invalid error for neuron ${i}: ${errors[i]}`);
            }
            neuron.updateWeights(errors[i], learningRate);
            return neuron.weights.reduce((sum, weight, j) => {
                if (!isValidNumber(weight) || !isValidNumber(errors[i])) {
                    throw new Error(`Invalid weight or error: weight=${weight}, error=${errors[i]}`);
                }
                return sum + weight * errors[i];
            }, 0);
        });
    }
}
