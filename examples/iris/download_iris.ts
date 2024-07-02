import * as https from 'https';
import * as fs from 'fs';
import * as path from 'path';
import * as zlib from 'zlib';
import * as stream from 'stream';
import * as util from 'util';
import * as zlib from 'zlib';

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

async function downloadAndUnzip() {
    try {
        const response = await new Promise((resolve, reject) => {
            https.get(url, resolve).on('error', reject);
        });

        await pipeline(response, fs.createWriteStream(zipFile));

        console.log('Download completed. Unzipping...');

        const zipContent = await fs.promises.readFile(zipFile);
        const unzipped = await new Promise((resolve, reject) => {
            zlib.unzip(zipContent, (err, buffer) => {
                if (err) reject(err);
                else resolve(buffer);
            });
        });

        await fs.promises.writeFile(outputFile, unzipped);

        console.log(`Iris dataset saved to: ${outputFile}`);

        // Clean up the zip file
        await fs.promises.unlink(zipFile);
    } catch (err) {
        console.error('Error downloading or unzipping the file:', err);
    }
}

downloadAndUnzip();
