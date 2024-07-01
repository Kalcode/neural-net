import { Neuron } from './Neuron';
import BigNumber from 'bignumber.js';
import { toBigNumber, isValidNumber } from './utils';

export class Layer {
    public neurons: Neuron[];
    private prevWeightDeltas: BigNumber[][];
    private prevBiasDeltas: BigNumber[];
    private invalidErrorCount: number;

    constructor(inputSize: number, outputSize: number) {
        this.neurons = Array.from({ length: outputSize }, () => new Neuron(inputSize));
        this.prevWeightDeltas = Array.from({ length: outputSize }, () => Array(inputSize).fill(toBigNumber(0)));
        this.prevBiasDeltas = Array(outputSize).fill(toBigNumber(0));
        this.invalidErrorCount = 0;
    }

    getInvalidErrorCount(): number {
        return this.invalidErrorCount;
    }

    resetInvalidErrorCount(): void {
        this.invalidErrorCount = 0;
    }

    forward(inputs: BigNumber[]): BigNumber[] {
        return this.neurons.map(neuron => toBigNumber(neuron.forward(inputs)));
    }

    train(errors: BigNumber[], learningRate: BigNumber, inputs: BigNumber[], momentum: BigNumber, batchSize: BigNumber, maxGradientNorm: BigNumber): void {
        if (errors.length !== this.neurons.length && this.neurons.length > 1) {
            errors = this.calculateHiddenLayerErrors(errors, inputs);
        }

        this.neurons.forEach((neuron, i) => {
            const { weightDeltas, biasDelta } = neuron.calculateGradients(errors[i], inputs);
            
            const newWeightDeltas = weightDeltas.map((delta, j) => delta.plus(momentum.times(this.prevWeightDeltas[i][j])));
            const newBiasDelta = biasDelta.plus(momentum.times(this.prevBiasDeltas[i]));

            const gradients = [...newWeightDeltas, newBiasDelta];
            const gradientNorm = toBigNumber(Math.sqrt(gradients.reduce((sum, g) => sum + g.pow(2).toNumber(), 0)));
            const scalingFactor = BigNumber.minimum(toBigNumber(1), maxGradientNorm.dividedBy(gradientNorm));

            neuron.applyGradients(newWeightDeltas.map(d => d.times(scalingFactor)), newBiasDelta.times(scalingFactor), learningRate);

            this.prevWeightDeltas[i] = newWeightDeltas;
            this.prevBiasDeltas[i] = newBiasDelta;
        });
    }

    private calculateHiddenLayerErrors(outputErrors: BigNumber[], inputs: BigNumber[]): BigNumber[] {
        const hiddenErrors: BigNumber[] = new Array(this.neurons.length).fill(toBigNumber(0));

        this.neurons.forEach((neuron, i) => {
            const output = neuron.forward(inputs);
            const derivative = output.times(toBigNumber(1).minus(output)); // Derivative of sigmoid
            
            outputErrors.forEach((outputError, j) => {
                hiddenErrors[i] = hiddenErrors[i].plus(outputError.times(this.neurons[j].weights[i]).times(derivative));
            });
        });

        return hiddenErrors;
    }
}
