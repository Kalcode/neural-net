import * as https from 'https';
import * as fs from 'fs';
import * as path from 'path';
import * as zlib from 'zlib';
import * as stream from 'stream';
import * as util from 'util';

const url = 'https://archive.ics.uci.edu/static/public/53/iris.zip';
const outputDir = path.join(__dirname, 'data');
const zipFile = path.join(outputDir, 'iris.zip');
const outputFile = path.join(outputDir, 'iris.data');

// Create the output directory if it doesn't exist
if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
}

console.log('Downloading Iris dataset...');

const pipeline = util.promisify(stream.pipeline);

async function downloadFile(url: string, outputPath: string): Promise<void> {
    const response = await new Promise<https.IncomingMessage>((resolve, reject) => {
        https.get(url, resolve).on('error', reject);
    });
    await pipeline(response, fs.createWriteStream(outputPath));
}

async function unzipFile(zipPath: string, outputPath: string): Promise<void> {
    const zipContent = await fs.promises.readFile(zipPath);
    const unzipped = await util.promisify(zlib.unzip)(zipContent);
    await fs.promises.writeFile(outputPath, unzipped);
}

async function cleanupFile(filePath: string): Promise<void> {
    try {
        await fs.promises.unlink(filePath);
    } catch (err) {
        console.warn(`Failed to delete file ${filePath}:`, err);
    }
}

async function downloadAndUnzip() {
    try {
        await downloadFile(url, zipFile);
        console.log('Download completed. Unzipping...');

        await unzipFile(zipFile, outputFile);
        console.log(`Iris dataset saved to: ${outputFile}`);

        await cleanupFile(zipFile);
    } catch (err) {
        console.error('Error downloading or unzipping the file:', err);
        throw err;
    }
}

downloadAndUnzip().catch(err => {
    console.error('Failed to download and unzip Iris dataset:', err);
    process.exit(1);
});
