const url = 'https://archive.ics.uci.edu/static/public/53/iris.zip';
const outputDir = import.meta.dir + '/data';
const zipFile = outputDir + '/iris.zip';
const outputFile = outputDir + '/iris.data';

// Create the output directory if it doesn't exist
await Bun.mkdir(outputDir, { recursive: true });

console.log('Downloading Iris dataset...');

async function downloadFile(url: string, outputPath: string): Promise<void> {
    const response = await fetch(url);
    await Bun.write(outputPath, response);
}

async function unzipFile(zipPath: string, outputPath: string): Promise<void> {
    const zipContent = await Bun.file(zipPath).arrayBuffer();
    const compressed = new Uint8Array(zipContent);
    const decompressed = Bun.gunzipSync(compressed);
    await Bun.write(outputPath, decompressed);
}

async function cleanupFile(filePath: string): Promise<void> {
    try {
        await Bun.file(filePath).remove();
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

await downloadAndUnzip().catch(err => {
    console.error('Failed to download and unzip Iris dataset:', err);
    process.exit(1);
});
