// Implementation tests

import { Layer } from "./Layer";
import { Neuron } from "./Neuron";

// Test a single neuron
console.log("Testing a single neuron:");
const neuron = new Neuron(3);
const neuronInputs = [0.5, 0.3, 0.2];
const neuronOutput = neuron.forward(neuronInputs);
console.log(`Neuron Inputs: ${neuronInputs}`);
console.log(`Neuron Output: ${neuronOutput}`);

console.log("\nTesting a layer of neurons:");
// Create a layer with 3 inputs and 2 outputs
const layer = new Layer(3, 2);
const layerInputs = [0.1, 0.2, 0.3];
const layerOutputs = layer.forward(layerInputs);
console.log(`Layer Inputs: ${layerInputs}`);
console.log(`Layer Outputs: ${layerOutputs}`);