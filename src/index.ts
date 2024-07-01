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

    constructor(inputNodes: number, hiddenNodes: number, outputNodes: number) {
        this.inputNodes = inputNodes;
        this.hiddenNodes = hiddenNodes;
        this.outputNodes = outputNodes;

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

    private dotProduct(a: number[], b: number[]): number {
        return a.reduce((sum, val, i) => sum + val * b[i], 0);
    }

    forward(input: number[]): number[] {
        // Calculate hidden layer
        const hidden = this.weightsIH.map((weights, i) =>
            this.sigmoid(this.dotProduct(weights, input) + this.biasH[i])
        );

        // Calculate output layer
        const output = this.weightsHO.map((weights, i) =>
            this.sigmoid(this.dotProduct(weights, hidden) + this.biasO[i])
        );

        return output;
    }
}

// Test the forward pass
const nn = new NeuralNetwork(2, 2, 1);
console.log(nn.forward([0, 0]));
console.log(nn.forward([0, 1]));
console.log(nn.forward([1, 0]));
console.log(nn.forward([1, 1]));
