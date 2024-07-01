import { Layer } from './Layer';

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
        return this.layers.reduce((prev, layer) => layer.forward(prev), inputs);
    }

    train(inputs: number[][], targets: number[][], epochs: number, learningRate: number): void {
        for (let epoch = 0; epoch < epochs; epoch++) {
            let totalError = 0;
            for (let i = 0; i < inputs.length; i++) {
                const output = this.forward(inputs[i]);
                if (output.some(isNaN)) {
                    console.error(`NaN output at epoch ${epoch}, input ${i}:`, output);
                    return;
                }
                const errors = targets[i].map((t, j) => t - output[j]);
                totalError += errors.reduce((sum, err) => sum + err * err, 0) / errors.length;

                for (let j = this.layers.length - 1; j >= 0; j--) {
                    this.layers[j].train(errors, learningRate, j === 0 ? inputs[i] : this.layers[j-1].forward(inputs[i]));
                }
            }
            const averageError = totalError / inputs.length;
            if (isNaN(averageError)) {
                console.error(`NaN average error at epoch ${epoch}`);
                return;
            }
            if (epoch % 100 === 0 || epoch === epochs - 1) {
                console.log(`Epoch ${epoch + 1}, Average Error: ${averageError}`);
            }
        }
    }
}
