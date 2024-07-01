/**
 * NeuralNetwork class represents a simple feedforward neural network
 * with one hidden layer. It can be used for various classification
 * and regression tasks, such as XOR and AND logic gates.
 */
export class NeuralNetwork {
    private inputNodes: number;
    private hiddenNodes: number;
    private outputNodes: number;
    private weightsIH: number[][]; // Weights from input to hidden layer
    private weightsHO: number[][]; // Weights from hidden to output layer
    private biasH: number[]; // Biases for hidden layer
    private biasO: number[]; // Biases for output layer
    private learningRate: number;

    /**
     * Constructor for the NeuralNetwork class.
     * @param inputNodes Number of input nodes
     * @param hiddenNodes Number of hidden nodes
     * @param outputNodes Number of output nodes
     * @param learningRate Learning rate for weight updates (default: 0.1)
     */
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

    /**
     * Initialize weights with random values between -1 and 1.
     * @param rows Number of rows in the weight matrix
     * @param cols Number of columns in the weight matrix
     * @returns A 2D array of initialized weights
     */
    private initializeWeights(rows: number, cols: number): number[][] {
        return Array.from({ length: rows }, () =>
            Array.from({ length: cols }, () => Math.random() * 2 - 1)
        );
    }

    /**
     * Initialize biases with random values between -1 and 1.
     * @param size Number of bias values to initialize
     * @returns An array of initialized bias values
     */
    private initializeBias(size: number): number[] {
        return Array.from({ length: size }, () => Math.random() * 2 - 1);
    }

    /**
     * Sigmoid activation function.
     * @param x Input value
     * @returns Sigmoid of the input
     */
    private sigmoid(x: number): number {
        return 1 / (1 + Math.exp(-x));
    }

    /**
     * Derivative of the sigmoid function.
     * @param x Input value (should be the output of a sigmoid function)
     * @returns Derivative of sigmoid at the input point
     */
    private sigmoidDerivative(x: number): number {
        return x * (1 - x);
    }

    /**
     * Compute the dot product of two vectors.
     * @param a First vector
     * @param b Second vector
     * @returns Dot product of a and b
     */
    private dotProduct(a: number[], b: number[]): number {
        return a.reduce((sum, val, i) => sum + val * b[i], 0);
    }

    /**
     * Perform a forward pass through the network.
     * @param input Input values
     * @returns Object containing hidden layer and output layer activations
     */
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

    /**
     * Train the neural network using backpropagation.
     * @param input Input values
     * @param target Target output values
     */
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

    /**
     * Calculate the mean squared error between predictions and targets.
     * @param predictions Predicted values
     * @param targets Target values
     * @returns Mean squared error
     */
    meanSquaredError(predictions: number[], targets: number[]): number {
        return predictions.reduce((sum, pred, i) => sum + Math.pow(targets[i] - pred, 2), 0) / predictions.length;
    }
}
