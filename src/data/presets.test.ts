import { describe, it, expect } from 'vitest';
import { SMILEY_PRESETS, SMILEY_LABELS, CLASS_PRESETS, CLASS_LABELS } from './presets';
import { CONFIG } from '../core/config';

const EXPECTED_SIZE = CONFIG.imageSize * CONFIG.imageSize; // 32*32 = 1024

describe('SMILEY_PRESETS', () => {
    it('contains 8 presets', () => {
        expect(SMILEY_PRESETS).toHaveLength(8);
    });

    it('each preset is a Float32Array', () => {
        for (const preset of SMILEY_PRESETS) {
            expect(preset).toBeInstanceOf(Float32Array);
        }
    });

    it('each preset has size imageSize*imageSize', () => {
        for (const preset of SMILEY_PRESETS) {
            expect(preset.length).toBe(EXPECTED_SIZE);
        }
    });

    it('all pixel values are in [-1, 1]', () => {
        for (const preset of SMILEY_PRESETS) {
            for (const v of preset) {
                expect(v).toBeGreaterThanOrEqual(-1.0);
                expect(v).toBeLessThanOrEqual(1.0);
            }
        }
    });

    it('different smiley variants are not identical (happy vs sad)', () => {
        const happy = SMILEY_PRESETS[0];  // happy
        const sad   = SMILEY_PRESETS[5];  // sad
        const diff = happy.reduce((acc, v, i) => acc + Math.abs(v - sad[i]), 0);
        expect(diff).toBeGreaterThan(0);
    });
});

describe('SMILEY_LABELS', () => {
    it('has same length as SMILEY_PRESETS', () => {
        expect(SMILEY_LABELS).toHaveLength(SMILEY_PRESETS.length);
    });
});

describe('CLASS_PRESETS', () => {
    it('has 3 classes (happy, neutral, sad)', () => {
        expect(CLASS_PRESETS).toHaveLength(3);
    });

    it('each class has at least one image', () => {
        for (const cls of CLASS_PRESETS) {
            expect(cls.length).toBeGreaterThan(0);
        }
    });

    it('each class image is Float32Array of correct size', () => {
        for (const cls of CLASS_PRESETS) {
            for (const img of cls) {
                expect(img).toBeInstanceOf(Float32Array);
                expect(img.length).toBe(EXPECTED_SIZE);
            }
        }
    });
});

describe('CLASS_LABELS', () => {
    it('has same length as CLASS_PRESETS', () => {
        expect(CLASS_LABELS).toHaveLength(CLASS_PRESETS.length);
    });
});
