import { Neuron } from './Neuron';
import { Layer } from './Layer';

// Log system info for debugging
console.log(`Node.js version: ${process.version}`);
console.log(`Bun version: ${Bun.version}`);
console.log(`V8 version: ${process.versions.v8}`);
console.log(`OS: ${process.platform} ${process.arch}`);
console.log(`Memory usage: ${process.memoryUsage().rss / 1024 / 1024} MB`);
console.log(`Current directory: ${process.cwd()}`);
console.log(`Command: ${process.argv.join(' ')}`);
console.log(`PID: ${process.pid}`);
console.log("\n");

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
