import { Layer } from './Layer';
import { isValidNumber } from './utils';

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

    forward(inputs: number[]): number[] {
        let currentInputs = inputs;
        for (const layer of this.layers) {
            currentInputs = layer.forward(currentInputs);
            if (currentInputs.some(output => !isValidNumber(output))) {
                console.error(`Invalid output from layer: ${currentInputs}`);
                return currentInputs.map(() => 0.5); // Default to middle of output range
            }
        }
        return currentInputs;
    }

    train(inputs: number[][], targets: number[][], epochs: number, learningRate: number): void {
        for (let epoch = 0; epoch < epochs; epoch++) {
            let totalError = 0;
            for (let i = 0; i < inputs.length; i++) {
                const output = this.forward(inputs[i]);
                if (output.some(val => !isValidNumber(val))) {
                    console.error(`Invalid output at epoch ${epoch}, input ${i}:`, inputs[i], "Output:", output);
                    return;
                }
                const errors = targets[i].map((t, j) => t - output[j]);
                const squaredErrors = errors.map(err => err * err);
                if (squaredErrors.some(val => !isValidNumber(val))) {
                    console.error(`Invalid squared errors at epoch ${epoch}, input ${i}:`, squaredErrors);
                    return;
                }
                totalError += squaredErrors.reduce((sum, err) => sum + err, 0) / errors.length;

                for (let j = this.layers.length - 1; j >= 0; j--) {
                    const layerInputs = j === 0 ? inputs[i] : this.layers[j-1].forward(inputs[i]);
                    this.layers[j].train(errors, learningRate, layerInputs);
                }
            }
            const averageError = totalError / inputs.length;
            if (!isValidNumber(averageError)) {
                console.error(`Invalid average error at epoch ${epoch}: ${averageError}`);
                return;
            }
            if (epoch % 100 === 0 || epoch === epochs - 1) {
                console.log(`Epoch ${epoch + 1}, Average Error: ${averageError}`);
            }
        }
    }
}
