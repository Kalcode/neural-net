import * as https from 'https';
import * as fs from 'fs';
import * as path from 'path';
import * as zlib from 'zlib';

const url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data';
const outputDir = path.join(__dirname, 'data');
const outputFile = path.join(outputDir, 'iris.data');

// Create the output directory if it doesn't exist
if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
}

console.log('Downloading Iris dataset...');

https.get(url, (response) => {
    const writeStream = fs.createWriteStream(outputFile);

    response.pipe(writeStream);

    writeStream.on('finish', () => {
        writeStream.close();
        console.log('Download completed.');
        console.log(`Iris dataset saved to: ${outputFile}`);
    });
}).on('error', (err) => {
    console.error('Error downloading the file:', err);
});
