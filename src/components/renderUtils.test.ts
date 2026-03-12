import { describe, it, expect, vi } from 'vitest';
import { renderImageToCanvas } from './renderUtils';

// ─── Mock canvas + context ────────────────────────────────────────────────────

function createMockContext(size = 4) {
    const pixels = new Uint8ClampedArray(size * size * 4);
    const imageData = { data: pixels, width: size, height: size };

    const ctx = {
        createImageData: vi.fn((_w: number, _h: number) => ({ ...imageData, data: new Uint8ClampedArray(size * size * 4) })),
        putImageData: vi.fn((imgData: { data: Uint8ClampedArray }) => {
            pixels.set(imgData.data);
        }),
    } as unknown as CanvasRenderingContext2D;

    return { ctx, pixels };
}

describe('renderImageToCanvas', () => {
    it('maps input -1.0 to pixel value 0', () => {
        const { ctx, pixels } = createMockContext(1);
        // Override createImageData to return a real structure with shared buffer
        const imgData = { data: new Uint8ClampedArray(4), width: 1, height: 1 };
        (ctx.createImageData as ReturnType<typeof vi.fn>).mockReturnValue(imgData);
        (ctx.putImageData as ReturnType<typeof vi.fn>).mockImplementation((d: typeof imgData) => {
            pixels.set(d.data);
        });

        renderImageToCanvas(new Float32Array([-1.0]), ctx);
        expect(pixels[0]).toBe(0); // R
        expect(pixels[3]).toBe(255); // A
    });

    it('maps input 1.0 to pixel value 255', () => {
        const { ctx, pixels } = createMockContext(1);
        const imgData = { data: new Uint8ClampedArray(4), width: 1, height: 1 };
        (ctx.createImageData as ReturnType<typeof vi.fn>).mockReturnValue(imgData);
        (ctx.putImageData as ReturnType<typeof vi.fn>).mockImplementation((d: typeof imgData) => {
            pixels.set(d.data);
        });

        renderImageToCanvas(new Float32Array([1.0]), ctx);
        expect(pixels[0]).toBe(255); // R
        expect(pixels[3]).toBe(255); // A
    });

    it('maps input 0.0 to pixel value approximately 127-128', () => {
        const { ctx, pixels } = createMockContext(1);
        const imgData = { data: new Uint8ClampedArray(4), width: 1, height: 1 };
        (ctx.createImageData as ReturnType<typeof vi.fn>).mockReturnValue(imgData);
        (ctx.putImageData as ReturnType<typeof vi.fn>).mockImplementation((d: typeof imgData) => {
            pixels.set(d.data);
        });

        renderImageToCanvas(new Float32Array([0.0]), ctx);
        expect(pixels[0]).toBeGreaterThanOrEqual(127);
        expect(pixels[0]).toBeLessThanOrEqual(128);
    });

    it('always sets alpha channel to 255', () => {
        const { ctx, pixels } = createMockContext(1);
        const imgData = { data: new Uint8ClampedArray(4), width: 1, height: 1 };
        (ctx.createImageData as ReturnType<typeof vi.fn>).mockReturnValue(imgData);
        (ctx.putImageData as ReturnType<typeof vi.fn>).mockImplementation((d: typeof imgData) => {
            pixels.set(d.data);
        });

        renderImageToCanvas(new Float32Array([0.5]), ctx);
        expect(pixels[3]).toBe(255);
    });

    it('clips values outside [-1, 1] gracefully', () => {
        const { ctx, pixels } = createMockContext(1);
        const imgData = { data: new Uint8ClampedArray(4), width: 1, height: 1 };
        (ctx.createImageData as ReturnType<typeof vi.fn>).mockReturnValue(imgData);
        (ctx.putImageData as ReturnType<typeof vi.fn>).mockImplementation((d: typeof imgData) => {
            pixels.set(d.data);
        });

        renderImageToCanvas(new Float32Array([5.0]), ctx);  // out of range
        expect(pixels[0]).toBe(255); // clamped to 255
    });
});
