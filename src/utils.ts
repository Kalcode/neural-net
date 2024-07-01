export function round(value: number, decimals: number = 6): number {
    return Number(Math.round(Number(value + 'e' + decimals)) + 'e-' + decimals);
}
