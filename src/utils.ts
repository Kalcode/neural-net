import BigNumber from 'bignumber.js';

export function isValidNumber(value: number | BigNumber): boolean {
    if (value instanceof BigNumber) {
        return !value.isNaN() && value.isFinite();
    }
    return typeof value === 'number' && isFinite(value) && !isNaN(value);
}

export function toBigNumber(value: number | string | BigNumber): BigNumber {
    return new BigNumber(value);
}

// console helpers
export const consoleHelpers = {
    systemInfo: () => {
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
    }
}
