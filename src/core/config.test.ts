import { describe, it, expect } from 'vitest';
import { CONFIG } from './config';

describe('CONFIG', () => {
    it('has imageSize of 32', () => {
        expect(CONFIG.imageSize).toBe(32);
    });

    it('has timesteps of 400', () => {
        expect(CONFIG.timesteps).toBe(400);
    });

    it('has betaStart less than betaEnd', () => {
        expect(CONFIG.betaStart).toBeLessThan(CONFIG.betaEnd);
    });

    it('has trainingSteps of 500 (pretrained model converges in 300-500)', () => {
        expect(CONFIG.trainingSteps).toBe(500);
    });

    it('has timeDim of 16 (sinusoidal embedding dimension)', () => {
        expect(CONFIG.timeDim).toBe(16);
    });

    it('has positive learning rate', () => {
        expect(CONFIG.learningRate).toBeGreaterThan(0);
        expect(CONFIG.learningRate).toBeLessThan(1);
    });

    it('has positive batchSize', () => {
        expect(CONFIG.batchSize).toBeGreaterThan(0);
    });
});
