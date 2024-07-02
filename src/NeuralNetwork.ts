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
    private clipValue: number; // Maximum allowed gradient value

    /**
     * Constructor for the NeuralNetwork class.
     * @param inputNodes Number of input nodes
     * @param hiddenNodes Number of hidden nodes
     * @param outputNodes Number of output nodes
     * @param learningRate Learning rate for weight updates (default: 0.1)
     */
    constructor(inputNodes: number, hiddenNodes: number, outputNodes: number, learningRate: number = 0.01) {                                           
        // add contructor values to instance
        this.inputNodes = inputNodes;                                                                                                                 
        this.hiddenNodes = hiddenNodes;                                                                                                               
        this.outputNodes = outputNodes;                                                                                                               
        this.learningRate = learningRate;
        this.clipValue = 1.0; // Set a reasonable clip value

        // Initialize weights and biases with random values
        this.weightsIH = this.initializeWeights(this.hiddenNodes, this.inputNodes);
        this.weightsHO = this.initializeWeights(this.outputNodes, this.hiddenNodes);
        this.biasH = this.initializeBias(this.hiddenNodes);
        this.biasO = this.initializeBias(this.outputNodes);
    }

    private clipGradients(gradients: number[][]): number[][] {
        return gradients.map(row => row.map(grad => Math.max(Math.min(grad, this.clipValue), -this.clipValue)));
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
        const calculateLayer = (weights: number[][], inputs: number[], biases: number[]): number[] =>
            weights.map((nodeWeights, i) =>
                this.sigmoid(this.dotProduct(nodeWeights, inputs) + biases[i])
            );

        const hidden = calculateLayer(this.weightsIH, input, this.biasH);
        const output = calculateLayer(this.weightsHO, hidden, this.biasO);

        return { hidden, output };
    }

    /**
     * Train the neural network using backpropagation.
     * @param input Input values
     * @param target Target output values
     */
    train(input: number[], target: number[]): number {
        const { hidden, output } = this.forward(input);

        // Calculate output layer errors
        const outputErrors = output.map((out, i) => target[i] - out);

        // Calculate hidden layer errors
        const hiddenErrors = this.weightsHO.reduce((errors, weights, i) => {
            weights.forEach((weight, j) => {
                errors[j] += outputErrors[i] * weight;
            });
            return errors;
        }, new Array(this.hiddenNodes).fill(0));

        // Calculate gradients
        const outputGradients = outputErrors.map((error, i) => error * this.sigmoidDerivative(output[i]));
        const hiddenGradients = hiddenErrors.map((error, i) => error * this.sigmoidDerivative(hidden[i]));

        // Clip gradients
        const clippedOutputGradients = this.clipGradients([outputGradients])[0];
        const clippedHiddenGradients = this.clipGradients([hiddenGradients])[0];

        // Update weights and biases for the output layer
        this.weightsHO.forEach((weights, i) => {
            weights.forEach((_, j) => {
                const delta = clippedOutputGradients[i] * hidden[j];
                this.weightsHO[i][j] += this.learningRate * delta;
            });
            this.biasO[i] += this.learningRate * clippedOutputGradients[i];
        });

        // Update weights and biases for the hidden layer
        this.weightsIH.forEach((weights, i) => {
            weights.forEach((_, j) => {
                const delta = clippedHiddenGradients[i] * input[j];
                this.weightsIH[i][j] += this.learningRate * delta;
            });
            this.biasH[i] += this.learningRate * clippedHiddenGradients[i];
        });

        // Calculate and return the mean squared error
        return outputErrors.reduce((sum, error) => sum + error * error, 0) / outputErrors.length;
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
