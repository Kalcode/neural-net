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
        const maxConsecutiveInvalidEpochs = 10;
        let consecutiveInvalidEpochs = 0;
        let bestAverageError = Infinity;
        let epochsSinceImprovement = 0;

        for (let epoch = 0; epoch < epochs; epoch++) {
            let totalError = 0;
            let validSamples = 0;
            let epochValid = true;

            for (let i = 0; i < inputs.length; i++) {
                const output = this.forward(inputs[i]);
                if (output.some(val => !isValidNumber(val))) {
                    console.error(`Invalid output at epoch ${epoch}, input ${i}:`, inputs[i], "Output:", output);
                    epochValid = false;
                    break;
                }

                const errors = targets[i].map((t, j) => {
                    const err = t - output[j];
                    if (!isValidNumber(err)) {
                        console.error(`Invalid error at epoch ${epoch}, input ${i}, output ${j}: ${err}`);
                        epochValid = false;
                        return 0;
                    }
                    return err;
                });

                if (!epochValid) break;

                const squaredErrors = errors.map(err => err * err);
                if (squaredErrors.some(val => !isValidNumber(val))) {
                    console.error(`Invalid squared errors at epoch ${epoch}, input ${i}:`, squaredErrors);
                    epochValid = false;
                    break;
                }

                totalError += squaredErrors.reduce((sum, err) => sum + err, 0) / errors.length;
                validSamples++;

                for (let j = this.layers.length - 1; j >= 0; j--) {
                    const layerInputs = j === 0 ? inputs[i] : this.layers[j-1].forward(inputs[i]);
                    this.layers[j].train(errors, learningRate, layerInputs);
                }
            }

            if (!epochValid || validSamples === 0) {
                console.error(`Invalid epoch ${epoch}: ${validSamples} valid samples out of ${inputs.length}`);
                consecutiveInvalidEpochs++;
                if (consecutiveInvalidEpochs >= maxConsecutiveInvalidEpochs) {
                    console.error(`Stopping training due to ${maxConsecutiveInvalidEpochs} consecutive invalid epochs`);
                    return;
                }
                continue;
            }

            consecutiveInvalidEpochs = 0;
            const averageError = totalError / validSamples;

            if (!isValidNumber(averageError)) {
                console.error(`Invalid average error at epoch ${epoch}: ${averageError}`);
                continue;
            }

            if (averageError < bestAverageError) {
                bestAverageError = averageError;
                epochsSinceImprovement = 0;
            } else {
                epochsSinceImprovement++;
            }

            if (epoch % 100 === 0 || epoch === epochs - 1) {
                console.log(`Epoch ${epoch + 1}, Average Error: ${averageError}, Valid Samples: ${validSamples}/${inputs.length}`);
            }

            // Early stopping condition
            if (epochsSinceImprovement >= 1000) {
                console.log(`Stopping early: No improvement for 1000 epochs. Best average error: ${bestAverageError}`);
                return;
            }

            // Stop if error is sufficiently low
            if (averageError < 0.001) {
                console.log(`Stopping early: Error threshold reached. Final average error: ${averageError}`);
                return;
            }
        }
    }
}
