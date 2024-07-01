import { Neuron } from './Neuron';

export class Layer {
    public neurons: Neuron[];

    constructor(inputSize: number, outputSize: number) {
        this.neurons = Array.from({ length: outputSize }, () => new Neuron(inputSize));
    }

    forward(inputs: number[]): number[] {
        return this.neurons.map(neuron => neuron.forward(inputs));
    }

    backpropagate(errors: number[], learningRate: number): number[] {
        return this.neurons.map((neuron, i) => {
            neuron.updateWeights(errors[i], learningRate);
            return neuron.weights.reduce((sum, weight, j) => sum + weight * errors[i], 0);
        });
    }
}
