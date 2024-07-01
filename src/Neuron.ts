export class Neuron {
    public weights: number[];
    public bias: number;

    constructor(inputSize: number) {
        this.weights = Array.from({ length: inputSize }, () => Math.random() * 2 - 1);
        this.bias = Math.random() * 2 - 1;
    }

    private sigmoid(x: number): number {
        if (x < -709) return 0;
        if (x > 709) return 1;
        return 1 / (1 + Math.exp(-x));
    }

    forward(inputs: number[]): number {
        if (inputs.length !== this.weights.length) {
            throw new Error("Input size does not match weight size");
        }

        const weightedSum = inputs.reduce((sum, input, i) => sum + input * this.weights[i], 0) + this.bias;
        return this.sigmoid(weightedSum);
    }

    updateWeights(error: number, learningRate: number, inputs: number[]): void {
        const output = this.forward(inputs);
        const delta = error * output * (1 - output);

        for (let i = 0; i < this.weights.length; i++) {
            this.weights[i] += learningRate * delta * inputs[i];
        }
        this.bias += learningRate * delta;
    }
}
