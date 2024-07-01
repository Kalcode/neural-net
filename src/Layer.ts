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
        console.log(`Layer training:`);
        console.log(`Errors:`, errors);
        console.log(`Learning rate:`, learningRate);
        console.log(`Inputs:`, inputs);
        this.neurons.forEach((neuron, i) => {
            console.log(`Training neuron ${i}:`);
            neuron.updateWeights(errors[i], learningRate, inputs);
        });
    }
}
