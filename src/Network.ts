import { Layer } from './Layer';
import { round, isValidNumber } from './utils';

export class Network {
    private layers: Layer[];

    constructor(layerSizes: number[]) {
        if (layerSizes.length < 2) {
            throw new Error("Network must have at least input and output layers");
        }
        this.layers = [];
        for (let i = 1; i < layerSizes.length; i++) {
            this.layers.push(new Layer(layerSizes[i-1], layerSizes[i]));
        }
    }

    get inputSize(): number {
        return this.layers[0].neurons[0].weights.length;
    }

    get outputSize(): number {
        return this.layers[this.layers.length - 1].neurons.length;
    }

    forward(inputs: number[]): number[] {
        return this.layers.reduce((prev, layer, index) => {
            try {
                const output = layer.forward(prev);
                if (output.some(v => !isValidNumber(v))) {
                    throw new Error(`Invalid layer output: ${output}`);
                }
                return output;
            } catch (error) {
                console.error(`Error in forward pass at layer ${index}:`, error);
                console.error('Input to layer:', prev);
                throw error;
            }
        }, inputs);
    }

    train(inputs: number[][], targets: number[][], epochs: number, learningRate: number): void {
        if (inputs.length !== targets.length) {
            throw new Error("Number of input samples must match number of target samples");
        }
        if (inputs[0].length !== this.inputSize) {
            throw new Error(`Input size mismatch. Expected ${this.inputSize}, got ${inputs[0].length}`);
        }
        if (targets[0].length !== this.outputSize) {
            throw new Error(`Output size mismatch. Expected ${this.outputSize}, got ${targets[0].length}`);
        }

        for (let epoch = 0; epoch < epochs; epoch++) {
            let totalError = 0;
            for (let i = 0; i < inputs.length; i++) {
                let output: number[] = [];
                let errors: number[] = [];
                let layerErrors: number[] = [];
                try {
                    output = this.forward(inputs[i]);
                    errors = targets[i].map((t, j) => {
                        if (!isValidNumber(t) || !isValidNumber(output[j])) {
                            throw new Error(`Invalid target or output: target=${t}, output=${output[j]}`);
                        }
                        return t - output[j];
                    });
                    totalError += errors.reduce((sum, err) => sum + err * err, 0) / errors.length;
                    
                    layerErrors = errors;
                    for (let j = this.layers.length - 1; j >= 0; j--) {
                        layerErrors = this.layers[j].backpropagate(layerErrors, learningRate);
                    }
                } catch (error) {
                    console.error(`Error in epoch ${epoch + 1}, input ${i}:`, error);
                    console.error('Current state:', { 
                        inputs: inputs[i], 
                        output: output, 
                        errors: errors, 
                        layerErrors: layerErrors 
                    });
                    console.error('Network state:', this.layers.map(layer => layer.neurons.map(neuron => ({
                        weights: neuron.weights,
                        bias: neuron.bias
                    }))));
                    console.error('Layer sizes:', this.layers.map(layer => layer.neurons.length));
                    console.error('Input size:', inputs[i].length);
                    console.error('Target size:', targets[i].length);
                    return; // Stop training if an error occurs
                }
            }
            const averageError = round(totalError / inputs.length);
            if (!isValidNumber(averageError)) {
                console.error(`Invalid average error: ${averageError}`);
                return; // Stop training if average error is invalid
            }
            if (epoch % 100 === 0 || epoch === epochs - 1) {
                console.log(`Epoch ${epoch + 1}, Average Error: ${averageError}`);
            }
        }
    }
}
