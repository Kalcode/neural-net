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
        if (errors.length !== this.neurons.length) {
            throw new Error(`Mismatch between errors length (${errors.length}) and neurons count (${this.neurons.length})`);
        }
        const nextErrors: number[] = new Array(this.neurons[0].weights.length).fill(0);
        this.neurons.forEach((neuron, i) => {
            const error = errors[i];
            if (!isValidNumber(error)) {
                throw new Error(`Invalid error for neuron ${i}: ${error}`);
            }
            const deltas = neuron.updateWeights(error, learningRate);
            deltas.forEach((delta, j) => {
                nextErrors[j] += delta;
            });
        });
        return nextErrors;
    }
}
