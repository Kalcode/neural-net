// Output is an xor input with the expected out
// Input: [0, 0] Output: 0
// Input: [0, 1] Output: 1
// Input: [1, 0] Output: 1
// Input: [1, 1] Output: 0

class NeuralNetwork {
    private inputNodes: number;
    private hiddenNodes: number;
    private outputNodes: number;
    private weightsIH: number[][];
    private weightsHO: number[][];
    private biasH: number[];
    private biasO: number[];
    private learningRate: number;

    constructor(inputNodes: number, hiddenNodes: number, outputNodes: number, learningRate: number = 0.1) {
        this.inputNodes = inputNodes;
        this.hiddenNodes = hiddenNodes;
        this.outputNodes = outputNodes;
        this.learningRate = learningRate;

        // Initialize weights and biases with random values
        this.weightsIH = this.initializeWeights(this.hiddenNodes, this.inputNodes);
        this.weightsHO = this.initializeWeights(this.outputNodes, this.hiddenNodes);
        this.biasH = this.initializeBias(this.hiddenNodes);
        this.biasO = this.initializeBias(this.outputNodes);
    }

    private initializeWeights(rows: number, cols: number): number[][] {
        return Array.from({ length: rows }, () =>
            Array.from({ length: cols }, () => Math.random() * 2 - 1)
        );
    }

    private initializeBias(size: number): number[] {
        return Array.from({ length: size }, () => Math.random() * 2 - 1);
    }

    private sigmoid(x: number): number {
        return 1 / (1 + Math.exp(-x));
    }

    private sigmoidDerivative(x: number): number {
        return x * (1 - x);
    }

    private dotProduct(a: number[], b: number[]): number {
        return a.reduce((sum, val, i) => sum + val * b[i], 0);
    }

    forward(input: number[]): { hidden: number[], output: number[] } {
        // Calculate hidden layer
        const hidden = this.weightsIH.map((weights, i) =>
            this.sigmoid(this.dotProduct(weights, input) + this.biasH[i])
        );

        // Calculate output layer
        const output = this.weightsHO.map((weights, i) =>
            this.sigmoid(this.dotProduct(weights, hidden) + this.biasO[i])
        );

        return { hidden, output };
    }

    train(input: number[], target: number[]): void {
        const { hidden, output } = this.forward(input);

        // Calculate output layer errors
        const outputErrors = output.map((out, i) => target[i] - out);

        // Update weights and biases for the output layer
        this.weightsHO.forEach((weights, i) => {
            weights.forEach((_, j) => {
                const delta = outputErrors[i] * this.sigmoidDerivative(output[i]) * hidden[j];
                this.weightsHO[i][j] += this.learningRate * delta;
            });
            this.biasO[i] += this.learningRate * outputErrors[i] * this.sigmoidDerivative(output[i]);
        });

        // Calculate hidden layer errors
        const hiddenErrors = this.weightsHO.reduce((errors, weights, i) => {
            weights.forEach((weight, j) => {
                errors[j] += outputErrors[i] * weight;
            });
            return errors;
        }, new Array(this.hiddenNodes).fill(0));

        // Update weights and biases for the hidden layer
        this.weightsIH.forEach((weights, i) => {
            weights.forEach((_, j) => {
                const delta = hiddenErrors[i] * this.sigmoidDerivative(hidden[i]) * input[j];
                this.weightsIH[i][j] += this.learningRate * delta;
            });
            this.biasH[i] += this.learningRate * hiddenErrors[i] * this.sigmoidDerivative(hidden[i]);
        });
    }

    meanSquaredError(predictions: number[], targets: number[]): number {
        return predictions.reduce((sum, pred, i) => sum + Math.pow(targets[i] - pred, 2), 0) / predictions.length;
    }
}

// XOR training data
const trainingData = [
    { input: [0, 0], target: [0] },
    { input: [0, 1], target: [1] },
    { input: [1, 0], target: [1] },
    { input: [1, 1], target: [0] }
];

// Create and train the neural network
const nn = new NeuralNetwork(2, 4, 1);
const epochs = 10000;

for (let i = 0; i < epochs; i++) {
    trainingData.forEach(data => {
        nn.train(data.input, data.target);
    });

    if (i % 1000 === 0) {
        const mse = trainingData.reduce((sum, data) => {
            const { output } = nn.forward(data.input);
            return sum + nn.meanSquaredError(output, data.target);
        }, 0) / trainingData.length;
        console.log(`Epoch ${i}: MSE = ${mse}`);
    }
}

// Test the trained network
console.log("Testing the trained network:");
trainingData.forEach(data => {
    const { output } = nn.forward(data.input);
    console.log(`Input: [${data.input}], Output: ${output[0].toFixed(4)}, Target: ${data.target[0]}`);
});
