import { runXORExample } from '../examples/xor';
import { runANDExample } from '../examples/and';
import { runORExample } from '../examples/or';
import { runNANDExample } from '../examples/nand';
import { runHalfAdderExample } from '../examples/half-adder';
import { runIrisExample } from '../examples/iris';
import { runLargestExample } from '../examples/largest';

console.log("Running XOR Example:");
runXORExample();

console.log("\n----------------------------\n");
console.log("Running AND Example:");
runANDExample();

console.log("\n----------------------------\n");
console.log("Running OR Example:");
runORExample();

console.log("\n----------------------------\n");
console.log("Running NAND Example:");
runNANDExample();

console.log("\n----------------------------\n");
console.log("Running Half Adder Example:");
runHalfAdderExample();

console.log("\n----------------------------\n");
console.log("Running Largest Example:");
runLargestExample();

console.log("\n----------------------------\n");
console.log("Running Iris Classification Example:");
runIrisExample();

