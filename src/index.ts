import { Neuron } from './Neuron';

// Create a neuron with 3 inputs
const neuron = new Neuron(3);

// Test the neuron with some sample inputs
const inputs = [0.5, 0.3, 0.2];
const output = neuron.forward(inputs);

console.log(`Inputs: ${inputs}`);
console.log(`Output: ${output}`);
