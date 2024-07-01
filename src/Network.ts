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
                return currentInputs.map(() => 0.5); // Default to middle of output range
            }
        }
        return currentInputs;
    }

    train(inputs: BigNumber[][], targets: BigNumber[][], epochs: number, learningRate: BigNumber, momentum: BigNumber, batchSize: BigNumber, maxGradientNorm: BigNumber): void {
        const maxConsecutiveInvalidEpochs = toBigNumber(10);
        let consecutiveInvalidEpochs = toBigNumber(0);
        let bestAverageError = toBigNumber(Infinity);
        let epochsSinceImprovement = toBigNumber(0);

        for (let epoch = 0; epoch < epochs; epoch++) {
            let totalError = toBigNumber(0);
            let validSamples = toBigNumber(0);
            let epochValid = true;

            for (let i = 0; i < inputs.length; i++) {
                const output = this.forward(inputs[i]);
                if (output.some(val => !isValidNumber(val))) {
                    console.error(`Invalid output at epoch ${epoch}, input ${i}:`, inputs[i], "Output:", output);
                    epochValid = false;
                    break;
                }

                console.log(`Epoch ${epoch}, Input ${i}:`);
                console.log(`Input:`, inputs[i]);
                console.log(`Output:`, output);
                console.log(`Target:`, targets[i]);

                const errors = targets[i].map((t, j) => {
                    const err = toBigNumber(t).minus(output[j]);
                    if (!isValidNumber(err)) {
                        console.error(`Invalid error at epoch ${epoch}, input ${i}, output ${j}: ${err.toString()}`);
                        epochValid = false;
                        return toBigNumber(0);
                    }
                    console.log(`Error for output ${j}: ${err.toString()}`);
                    return err;
                });

                if (!epochValid) break;

                console.log(`Errors:`, errors);

                const squaredErrors = errors.map(err => err.pow(2));
                if (squaredErrors.some(val => !isValidNumber(val))) {
                    console.error(`Invalid squared errors at epoch ${epoch}, input ${i}:`, squaredErrors.map(e => e.toString()));
                    epochValid = false;
                    break;
                }

                totalError = totalError.plus(squaredErrors.reduce((sum, err) => sum.plus(err), toBigNumber(0)).dividedBy(toBigNumber(errors.length)));
                validSamples = validSamples.plus(1);

                for (let j = this.layers.length - 1; j >= 0; j--) {
                    const layerInputs = j === 0 ? inputs[i] : this.layers[j-1].forward(inputs[i]);
                    console.log(`Training layer ${j}:`);
                    console.log(`Layer inputs:`, layerInputs);
                    this.layers[j].train(errors, learningRate, layerInputs, momentum, batchSize, maxGradientNorm);
                }
            }

            if (!epochValid || validSamples.isEqualTo(0)) {
                console.error(`Invalid epoch ${epoch}: ${validSamples.toString()} valid samples out of ${inputs.length}`);
                consecutiveInvalidEpochs = consecutiveInvalidEpochs.plus(1);
                if (consecutiveInvalidEpochs.isGreaterThanOrEqualTo(maxConsecutiveInvalidEpochs)) {
                    console.error(`Stopping training due to ${maxConsecutiveInvalidEpochs.toString()} consecutive invalid epochs`);
                    return;
                }
                continue;
            }

            consecutiveInvalidEpochs = toBigNumber(0);
            const averageError = totalError.dividedBy(validSamples);

            if (!isValidNumber(averageError)) {
                console.error(`Invalid average error at epoch ${epoch}: ${averageError}`);
                continue;
            }

            if (averageError.isLessThan(bestAverageError)) {
                bestAverageError = averageError;
                epochsSinceImprovement = toBigNumber(0);
            } else {
                epochsSinceImprovement = epochsSinceImprovement.plus(1);
            }

            if (epoch % 100 === 0 || epoch === epochs - 1) {
                console.log(`Epoch ${epoch + 1}, Average Error: ${averageError.toString()}, Valid Samples: ${validSamples.toString()}/${inputs.length}`);
            }

            // Early stopping condition
            if (epochsSinceImprovement.isGreaterThanOrEqualTo(1000)) {
                console.log(`Stopping early: No improvement for 1000 epochs. Best average error: ${bestAverageError.toString()}`);
                return;
            }

            // Stop if error is sufficiently low
            if (averageError.isLessThan(0.001)) {
                console.log(`Stopping early: Error threshold reached. Final average error: ${averageError.toString()}`);
                return;
            }
        }
    }
}
