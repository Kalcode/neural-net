import { Neuron } from './Neuron';
import { isValidNumber } from './utils';

export class Layer {
    public neurons: Neuron[];
    private prevWeightDeltas: number[][];
    private prevBiasDeltas: number[];
    private invalidErrorCount: number;

    constructor(inputSize: number, outputSize: number) {
        this.neurons = Array.from({ length: outputSize }, () => new Neuron(inputSize));
        this.prevWeightDeltas = Array.from({ length: outputSize }, () => Array(inputSize).fill(0));
        this.prevBiasDeltas = Array(outputSize).fill(0);
        this.invalidErrorCount = 0;
    }

    getInvalidErrorCount(): number {
        return this.invalidErrorCount;
    }

    resetInvalidErrorCount(): void {
        this.invalidErrorCount = 0;
    }

    forward(inputs: number[]): number[] {
        return this.neurons.map(neuron => neuron.forward(inputs));
    }

    train(errors: number[], learningRate: number, inputs: number[], momentum: number, batchSize: number, maxGradientNorm: number): void {
        if (errors.length !== this.neurons.length && this.neurons.length > 1) {
            errors = this.calculateHiddenLayerErrors(errors, inputs);
        }

        this.neurons.forEach((neuron, i) => {
            const { weightDeltas, biasDelta } = neuron.calculateGradients(errors[i], inputs);
            
            const newWeightDeltas = weightDeltas.map((delta, j) => delta + momentum * this.prevWeightDeltas[i][j]);
            const newBiasDelta = biasDelta + momentum * this.prevBiasDeltas[i];

            const gradients = [...newWeightDeltas, newBiasDelta];
            const gradientNorm = Math.sqrt(gradients.reduce((sum, g) => sum + g ** 2, 0));
            const scalingFactor = Math.min(1, maxGradientNorm / gradientNorm);

            neuron.applyGradients(newWeightDeltas.map(d => d * scalingFactor), newBiasDelta * scalingFactor, learningRate);

            this.prevWeightDeltas[i] = newWeightDeltas;
            this.prevBiasDeltas[i] = newBiasDelta;
        });
    }

    private calculateHiddenLayerErrors(outputErrors: number[], inputs: number[]): number[] {
        const hiddenErrors: number[] = new Array(this.neurons.length).fill(0);

        this.neurons.forEach((neuron, i) => {
            const output = neuron.forward(inputs);
            const derivative = output * (1 - output); // Derivative of sigmoid
            
            outputErrors.forEach((outputError, j) => {
                hiddenErrors[i] += outputError * this.neurons[j].weights[i] * derivative;
            });
        });

        return hiddenErrors;
    }
}
