import { Layer } from './Layer';
import { round } from './utils';

export class Network {
    private layers: Layer[];

    constructor(layerSizes: number[]) {
        this.layers = [];
        for (let i = 1; i < layerSizes.length; i++) {
            this.layers.push(new Layer(layerSizes[i-1], layerSizes[i]));
        }
    }

    forward(inputs: number[]): number[] {
        return this.layers.reduce((prev, layer) => layer.forward(prev), inputs);
    }

    train(inputs: number[][], targets: number[][], epochs: number, learningRate: number): void {
        for (let epoch = 0; epoch < epochs; epoch++) {
            let totalError = 0;
            for (let i = 0; i < inputs.length; i++) {
                const output = this.forward(inputs[i]);
                const errors = targets[i].map((t, j) => t - output[j]);
                totalError += errors.reduce((sum, err) => sum + err * err, 0) / errors.length;
                
                let layerErrors = errors;
                for (let j = this.layers.length - 1; j >= 0; j--) {
                    layerErrors = this.layers[j].backpropagate(layerErrors, learningRate);
                }
            }
            console.log(`Epoch ${epoch + 1}, Average Error: ${round(totalError / inputs.length)}`);
        }
    }
}
