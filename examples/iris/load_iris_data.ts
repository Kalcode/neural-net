import * as fs from 'fs';
import * as path from 'path';

import type { IrisFeatures, IrisClass } from './iris_data';

interface IrisDataPoint {
    features: IrisFeatures;
    class: IrisClass;
}

export function loadIrisData(): IrisDataPoint[] {
    const dataPath = path.join(__dirname, 'data', 'iris.data');
    console.log(`Loading Iris data from: ${dataPath}`);
    const data = fs.readFileSync(dataPath, 'utf8');
    const lines = data.split('\n').filter(line => line.trim() !== '');
    console.log(`Found ${lines.length} data points`);

    const irisData = lines.map((line, index) => {
        const [sepalLength, sepalWidth, petalLength, petalWidth, className] = line.split(',');
        if (!sepalLength || !sepalWidth || !petalLength || !petalWidth || !className) {
            console.error(`Invalid data line at index ${index}: ${line}`);
            return null;
        }
        return {
            features: {
                sepalLength: parseFloat(sepalLength),
                sepalWidth: parseFloat(sepalWidth),
                petalLength: parseFloat(petalLength),
                petalWidth: parseFloat(petalWidth)
            },
            class: className.trim() as IrisClass
        };
    }).filter((dataPoint): dataPoint is IrisDataPoint => dataPoint !== null);

    console.log(`Successfully loaded ${irisData.length} valid data points`);
    return irisData;
}
