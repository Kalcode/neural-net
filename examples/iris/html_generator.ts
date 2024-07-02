import type { ConfusionMatrix } from './metrics';

export function generateHTML(
    confusionMatrix: ConfusionMatrix,
    metrics: { [key: string]: { precision: number; recall: number; f1Score: number } },
    learningCurve: number[],
    accuracy: number
): string {
    const classes = Object.keys(confusionMatrix);

    const confusionMatrixHTML = `
        <h2>Confusion Matrix</h2>
        <table border="1">
            <tr>
                <th></th>
                ${classes.map(cls => `<th>${cls}</th>`).join('')}
            </tr>
            ${classes.map(actualClass => `
                <tr>
                    <th>${actualClass}</th>
                    ${classes.map(predictedClass => `
                        <td>${confusionMatrix[actualClass][predictedClass]}</td>
                    `).join('')}
                </tr>
            `).join('')}
        </table>
    `;

    const metricsHTML = `
        <h2>Metrics</h2>
        <table border="1">
            <tr>
                <th>Class</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-Score</th>
            </tr>
            ${Object.entries(metrics).map(([cls, { precision, recall, f1Score }]) => `
                <tr>
                    <td>${cls}</td>
                    <td>${precision.toFixed(4)}</td>
                    <td>${recall.toFixed(4)}</td>
                    <td>${f1Score.toFixed(4)}</td>
                </tr>
            `).join('')}
        </table>
    `;

    const learningCurveHTML = `
        <h2>Learning Curve</h2>
        <div id="learningCurve" style="width: 600px; height: 400px;"></div>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <script>
            var trace = {
                x: ${JSON.stringify(Array.from({ length: learningCurve.length }, (_, i) => i))},
                y: ${JSON.stringify(learningCurve)},
                type: 'scatter'
            };
            var layout = {
                title: 'Learning Curve',
                xaxis: { title: 'Epoch' },
                yaxis: { title: 'Mean Squared Error' }
            };
            Plotly.newPlot('learningCurve', [trace], layout);
        </script>
    `;

    return `
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Iris Classification Results</title>
        </head>
        <body>
            <h1>Iris Classification Results</h1>
            <p>Overall Accuracy: ${accuracy.toFixed(2)}%</p>
            ${confusionMatrixHTML}
            ${metricsHTML}
            ${learningCurveHTML}
        </body>
        </html>
    `;
}
