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

    train(inputs: number[][], targets: number[][], epochs: number, learningRate: number, momentum: number, batchSize: number, maxGradientNorm: number): void {
        let bestAverageError = Infinity;
        let epochsSinceImprovement = 0;

        for (let epoch = 0; epoch < epochs; epoch++) {
            let totalError = 0;

            for (let i = 0; i < inputs.length; i += batchSize) {
                const batchInputs = inputs.slice(i, i + batchSize);
                const batchTargets = targets.slice(i, i + batchSize);

                const batchErrors = batchInputs.map((input, j) => {
                    const output = this.forward(input);
                    return batchTargets[j].map((t, k) => t - output[k]);
                });

                const batchError = batchErrors.reduce((sum, errors) => sum + errors.reduce((s, e) => s + e ** 2, 0), 0) / batchErrors.length;
                totalError += batchError;

                for (let j = this.layers.length - 1; j >= 0; j--) {
                    const layerInputs = j === 0 ? batchInputs : this.layers[j-1].forward(batchInputs);
                    this.layers[j].train(batchErrors.flat(), learningRate, layerInputs, momentum, batchSize, maxGradientNorm);
                }
            }

            const averageError = totalError / (inputs.length / batchSize);

            if (averageError < bestAverageError) {
                bestAverageError = averageError;
                epochsSinceImprovement = 0;
            } else {
                epochsSinceImprovement++;
            }

            console.log(`Epoch ${epoch + 1}, Average Error: ${averageError}`);

            // Early stopping conditions
            if (epochsSinceImprovement >= 100) {
                console.log(`Stopping early: No improvement for 100 epochs. Best average error: ${bestAverageError}`);
                return;
            }

            if (averageError < 0.001) {
                console.log(`Stopping early: Error threshold reached. Final average error: ${averageError}`);
                return;
            }
        }
    }
}
