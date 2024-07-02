export interface IrisFeatures {
    sepalLength: number;
    sepalWidth: number;
    petalLength: number;
    petalWidth: number;
}

export type IrisClass = 'Iris-setosa' | 'Iris-versicolor' | 'Iris-virginica';

export const irisData: Array<{ features: IrisFeatures; class: IrisClass }> = [
    { features: { sepalLength: 5.1, sepalWidth: 3.5, petalLength: 1.4, petalWidth: 0.2 }, class: 'setosa' },
    { features: { sepalLength: 4.9, sepalWidth: 3.0, petalLength: 1.4, petalWidth: 0.2 }, class: 'setosa' },
    { features: { sepalLength: 7.0, sepalWidth: 3.2, petalLength: 4.7, petalWidth: 1.4 }, class: 'versicolor' },
    { features: { sepalLength: 6.4, sepalWidth: 3.2, petalLength: 4.5, petalWidth: 1.5 }, class: 'versicolor' },
    { features: { sepalLength: 6.3, sepalWidth: 3.3, petalLength: 6.0, petalWidth: 2.5 }, class: 'virginica' },
    { features: { sepalLength: 5.8, sepalWidth: 2.7, petalLength: 5.1, petalWidth: 1.9 }, class: 'virginica' },
];

export function featuresToArray(features: IrisFeatures): number[] {
    return [features.sepalLength, features.sepalWidth, features.petalLength, features.petalWidth];
}

export function classToArray(irisClass: IrisClass): number[] {
    switch (irisClass) {
        case 'Iris-setosa': return [1, 0, 0];
        case 'Iris-versicolor': return [0, 1, 0];
        case 'Iris-virginica': return [0, 0, 1];
        default:
            console.error(`Invalid iris class: ${irisClass}`);
            return [0, 0, 0];
    }
}

export function arrayToClass(output: number[]): IrisClass {
    const maxIndex = output.indexOf(Math.max(...output));
    switch (maxIndex) {
        case 0: return 'setosa';
        case 1: return 'versicolor';
        case 2: return 'virginica';
        default: throw new Error('Invalid output array');
    }
}
