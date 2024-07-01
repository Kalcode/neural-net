import { Layer } from './Layer';
import { isValidNumber, toBigNumber } from './utils';
import BigNumber from 'bignumber.js';

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

    forward(inputs: BigNumber[]): BigNumber[] {
        let currentInputs = inputs.map(toBigNumber);
        for (const layer of this.layers) {
            currentInputs = layer.forward(currentInputs);
            if (currentInputs.some(output => !isValidNumber(output))) {
                console.error(`Invalid output from layer: ${currentInputs}`);
                return currentInputs.map(() => toBigNumber(0.5)); // Default to middle of output range
            }
        }
        return currentInputs.map(toBigNumber);
    }

    train(inputs: BigNumber[][], targets: BigNumber[][], epochs: number, learningRate: BigNumber, momentum: BigNumber, batchSize: BigNumber, maxGradientNorm: BigNumber): void {
        const maxConsecutiveInvalidEpochs = 10;
        let consecutiveInvalidEpochs = 0;
        let bestAverageError = Infinity;
        let epochsSinceImprovement = 0;

        for (let epoch = 0; epoch < epochs; epoch++) {
            let totalError = 0;
            let validSamples = 0;

            for (let i = 0; i < inputs.length; i++) {
                const output = this.forward(inputs[i]);
                const errors = targets[i].map((t, j) => t.minus(output[j]));

                totalError += errors.reduce((sum, err) => sum + err.pow(2).toNumber(), 0) / errors.length;
                validSamples++;

                for (let j = this.layers.length - 1; j >= 0; j--) {
                    const layerInputs = j === 0 ? inputs[i] : this.layers[j-1].forward(inputs[i]);
                    this.layers[j].train(errors, learningRate, layerInputs, momentum, batchSize, maxGradientNorm);
                }
            }

            const averageError = totalError / validSamples;

            if (averageError < bestAverageError) {
                bestAverageError = averageError;
                epochsSinceImprovement = 0;
            } else {
                epochsSinceImprovement++;
            }

            console.log(`Epoch ${epoch + 1}, Average Error: ${averageError}, Valid Samples: ${validSamples}/${inputs.length}`);

            // Early stopping conditions
            if (epochsSinceImprovement >= 1000) {
                console.log(`Stopping early: No improvement for 1000 epochs. Best average error: ${bestAverageError}`);
                return;
            }

            if (averageError < 0.001) {
                console.log(`Stopping early: Error threshold reached. Final average error: ${averageError}`);
                return;
            }
        }
    }
}
