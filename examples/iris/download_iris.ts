import { mkdir, writeFile } from 'node:fs/promises';
import { createWriteStream } from 'node:fs';
import { pipeline } from 'node:stream/promises';
import { Readable } from 'node:stream';
import { Extract } from 'unzipper';

const url = 'https://archive.ics.uci.edu/static/public/53/iris.zip';
const outputDir = import.meta.dir + '/data';
const outputFile = outputDir + '/iris.data';

async function downloadAndExtract(url: string, outputPath: string): Promise<void> {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    if (!response.body) throw new Error('Response body is null');

    await pipeline(
        Readable.fromWeb(response.body as ReadableStream<Uint8Array>),
        Extract({ path: outputDir })
    );

    console.log('Iris dataset downloaded and extracted successfully!');
}

async function main() {
    try {
        await mkdir(outputDir, { recursive: true });
        await downloadAndExtract(url, outputFile);
    } catch (error) {
        console.error('Error processing Iris dataset:', error);
        process.exit(1);
    }
}

await main();