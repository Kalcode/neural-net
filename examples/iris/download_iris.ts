import { mkdir, unlink, writeFile } from 'node:fs/promises';
import { createGunzip } from 'node:zlib';
import { pipeline } from 'node:stream/promises';
import { createWriteStream } from 'node:fs';
import fetch from 'node-fetch';

const url = 'https://archive.ics.uci.edu/static/public/53/iris.zip';
const outputDir = import.meta.dir + '/data';
const outputFile = outputDir + '/iris.data';

// Create the output directory if it doesn't exist
await mkdir(outputDir, { recursive: true });

console.log('Downloading Iris dataset...');

async function downloadFile(url: string, outputPath: string): Promise<void> {
    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        if (!response.body) throw new Error('Response body is null');
        await pipeline(
            response.body,
            createGunzip(),
            createWriteStream(outputPath)
        );
    } catch (error) {
        console.error(`Error downloading and extracting file from ${url} to ${outputPath}:`, error);
        throw error;
    }
}

async function cleanupFile(filePath: string): Promise<void> {
    try {
        await unlink(filePath);
    } catch (err) {
        console.warn(`Failed to delete file ${filePath}:`, err);
    }
}

async function main() {
    try {
        await mkdir(outputDir, { recursive: true });
        await downloadFile(url, outputFile);
        console.log('Iris dataset downloaded and extracted successfully!');
    } catch (error) {
        console.error('Error processing Iris dataset:', error);
        process.exit(1);
    }
}

await main();
