import * as fs from 'fs';
import * as path from 'path';
import { IrisFeatures, IrisClass } from './iris_data';

interface IrisDataPoint {
    features: IrisFeatures;
    class: IrisClass;
}

export function loadIrisData(): IrisDataPoint[] {
    const dataPath = path.join(__dirname, 'data', 'iris.data');
    const data = fs.readFileSync(dataPath, 'utf8');
    const lines = data.split('\n').filter(line => line.trim() !== '');

    return lines.map(line => {
        const [sepalLength, sepalWidth, petalLength, petalWidth, className] = line.split(',');
        return {
            features: {
                sepalLength: parseFloat(sepalLength),
                sepalWidth: parseFloat(sepalWidth),
                petalLength: parseFloat(petalLength),
                petalWidth: parseFloat(petalWidth)
            },
            class: className.trim() as IrisClass
        };
    });
}
