import { Neuron } from './Neuron';

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

// Create a neuron with 3 inputs
const neuron = new Neuron(3);

// Test the neuron with some sample inputs
const inputs = [0.5, 0.3, 0.2];
const output = neuron.forward(inputs);

console.log(`Inputs: ${inputs}`);
console.log(`Output: ${output}`);
