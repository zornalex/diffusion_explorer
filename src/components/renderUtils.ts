import { CONFIG } from '../core/config';

/**
 * Renders a Float32Array image (range [-1, 1]) onto a canvas context.
 * @param data - pixel values in [-1, 1]
 * @param ctx  - target 2D context (canvas must already be sized CONFIG.imageSize)
 */
export function renderImageToCanvas(data: Float32Array, ctx: CanvasRenderingContext2D): void {
    const imgData = ctx.createImageData(CONFIG.imageSize, CONFIG.imageSize);
    const out = imgData.data;
    for (let i = 0; i < data.length; i++) {
        const val = Math.max(0, Math.min(255, (data[i] + 1.0) * 127.5));
        out[i * 4] = val;
        out[i * 4 + 1] = val;
        out[i * 4 + 2] = val;
        out[i * 4 + 3] = 255;
    }
    ctx.putImageData(imgData, 0, 0);
}

/**
 * Renders a noise tensor (range roughly [-3, 3]) onto a canvas context.
 */
export function renderNoiseToCanvas(data: Float32Array, ctx: CanvasRenderingContext2D): void {
    const imgData = ctx.createImageData(CONFIG.imageSize, CONFIG.imageSize);
    const out = imgData.data;
    for (let i = 0; i < data.length; i++) {
        const val = Math.max(0, Math.min(255, (data[i] + 3) / 6 * 255));
        out[i * 4] = val;
        out[i * 4 + 1] = val;
        out[i * 4 + 2] = val;
        out[i * 4 + 3] = 255;
    }
    ctx.putImageData(imgData, 0, 0);
}

/**
 * Creates and sizes a canvas element, returning its context.
 */
export function setupCanvas(
    id: string,
    internalSize: number,
    displaySize: string
): CanvasRenderingContext2D {
    const canvas = document.getElementById(id) as HTMLCanvasElement;
    canvas.width = internalSize;
    canvas.height = internalSize;
    canvas.style.width = displaySize;
    canvas.style.height = displaySize;
    canvas.style.imageRendering = 'pixelated';
    return canvas.getContext('2d')!;
}
