import { describe, it, expect } from 'vitest';
import { DiffusionModel } from './diffusion';
import { CONFIG } from './config';

describe('DiffusionModel', () => {
    const model = new DiffusionModel();
    const T = CONFIG.timesteps;

    it('returns correct timestep count', () => {
        expect(model.getTimesteps()).toBe(T);
    });

    it('alphaBar at t<0 returns 1 (boundary)', () => {
        expect(model.getAlphaBar(-1)).toBe(1);
    });

    it('alphaBar is close to 1 at t=1 (first step)', () => {
        const abar1 = model.getAlphaBar(0); // 0-indexed: t=1 in paper
        expect(abar1).toBeGreaterThan(0.98);
        expect(abar1).toBeLessThanOrEqual(1);
    });

    it('alphaBar is close to 0 at t=T (full noise)', () => {
        const abarT = model.getAlphaBar(T - 1);
        expect(abarT).toBeGreaterThanOrEqual(0);
        expect(abarT).toBeLessThan(0.05);
    });

    it('alphaBar is monotonically decreasing', () => {
        let prev = model.getAlphaBar(0);
        for (let t = 1; t < T; t++) {
            const curr = model.getAlphaBar(t);
            expect(curr).toBeLessThan(prev);
            prev = curr;
        }
    });

    it('getAlpha at t<0 returns 1', () => {
        expect(model.getAlpha(-1)).toBe(1);
    });

    it('all getAlpha values are in (0, 1)', () => {
        for (let t = 0; t < T; t++) {
            const a = model.getAlpha(t);
            expect(a).toBeGreaterThan(0);
            expect(a).toBeLessThan(1);
        }
    });

    it('addNoise at t=0 returns original image unchanged', () => {
        const img = new Float32Array([0.5, -0.5, 1.0, -1.0, 0.0]);
        const result = model.addNoise(img, 0);
        expect(Array.from(result)).toEqual(Array.from(img));
    });

    it('addNoiseWithEpsilon is deterministic given same epsilon', () => {
        const img = new Float32Array(32 * 32).fill(0.5);
        const eps = new Float32Array(32 * 32).fill(0.3);
        const r1 = model.addNoiseWithEpsilon(img, 50, eps);
        const r2 = model.addNoiseWithEpsilon(img, 50, eps);
        expect(Array.from(r1)).toEqual(Array.from(r2));
    });

    it('addNoiseWithEpsilon at t=0 returns original', () => {
        const img = new Float32Array([1.0, -1.0, 0.5]);
        const eps = new Float32Array([0.1, 0.2, 0.3]);
        const result = model.addNoiseWithEpsilon(img, 0, eps);
        expect(Array.from(result)).toEqual(Array.from(img));
    });

    it('addNoise at high t has high variance (more noise than signal)', () => {
        const img = new Float32Array(100).fill(1.0); // all white
        const noisy = model.addNoise(img, T - 1);
        // With near-zero alphaBar, most signal should be gone; check mean is not close to 1
        const mean = noisy.reduce((a, b) => a + b, 0) / noisy.length;
        expect(Math.abs(mean)).toBeLessThan(0.7); // signal mostly lost
    });
});
