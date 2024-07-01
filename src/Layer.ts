import { Neuron } from './Neuron';

export class Layer {
    public neurons: Neuron[];

    constructor(inputSize: number, outputSize: number) {
        this.neurons = Array.from({ length: outputSize }, () => new Neuron(inputSize));
    }

    forward(inputs: number[]): number[] {
        return this.neurons.map(neuron => neuron.forward(inputs));
    }

    train(errors: number[], learningRate: number, inputs: number[]): void {
        this.neurons.forEach((neuron, i) => {
            neuron.updateWeights(errors[i], learningRate, inputs);
        });
    }
}
