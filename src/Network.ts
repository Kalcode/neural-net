import { Layer } from './Layer';
import { round, isValidNumber } from './utils';

export class Network {
    private layers: Layer[];

    constructor(layerSizes: number[]) {
        this.layers = [];
        for (let i = 1; i < layerSizes.length; i++) {
            this.layers.push(new Layer(layerSizes[i-1], layerSizes[i]));
        }
    }

    forward(inputs: number[]): number[] {
        return this.layers.reduce((prev, layer) => {
            const output = layer.forward(prev);
            if (output.some(v => !isValidNumber(v))) {
                throw new Error(`Invalid layer output: ${output}`);
            }
            return output;
        }, inputs);
    }

    train(inputs: number[][], targets: number[][], epochs: number, learningRate: number): void {
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
                        console.log(`Before backpropagation - Layer ${j}, Errors:`, layerErrors);
                        if (layerErrors.some(err => !isValidNumber(err))) {
                            throw new Error(`Invalid errors before backpropagation: ${layerErrors}`);
                        }
                        layerErrors = this.layers[j].backpropagate(layerErrors, learningRate);
                        console.log(`After backpropagation - Layer ${j}, New Errors:`, layerErrors);
                        if (layerErrors.some(err => !isValidNumber(err))) {
                            throw new Error(`Invalid errors after backpropagation: ${layerErrors}`);
                        }
                    }
                } catch (error) {
                    console.error(`Error in epoch ${epoch + 1}, input ${i}:`, error);
                    console.error('Current state:', { 
                        inputs: inputs[i], 
                        output: output, 
                        errors: errors, 
                        layerErrors: layerErrors 
                    });
                    return; // Stop training if an error occurs
                }
            }
            const averageError = round(totalError / inputs.length);
            if (!isValidNumber(averageError)) {
                console.error(`Invalid average error: ${averageError}`);
                return; // Stop training if average error is invalid
            }
            console.log(`Epoch ${epoch + 1}, Average Error: ${averageError}`);
        }
    }
}
