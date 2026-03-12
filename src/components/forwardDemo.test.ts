import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ForwardProcessDemo } from './forwardDemo';
import { DiffusionModel } from '../core/diffusion';

// Mock canvas utilities — happy-dom does not support canvas rendering in CI
vi.mock('./renderUtils', () => ({
    setupCanvas: (_id: string, _size: number, _display: string) => {
        // Return a minimal fake CanvasRenderingContext2D
        return {
            canvas: { width: 32, height: 32 },
            fillStyle: '',
            font: '',
            textAlign: '',
            textBaseline: '',
            fillRect: () => undefined,
            fillText: () => undefined,
            putImageData: () => undefined,
            createImageData: (w: number, h: number) => ({
                data: new Uint8ClampedArray(w * h * 4),
                width: w,
                height: h,
            }),
        } as unknown as CanvasRenderingContext2D;
    },
    renderImageToCanvas: () => undefined,
}));

// Mock requestAnimationFrame (happy-dom stubs it but we want control)
beforeEach(() => {
    vi.stubGlobal('requestAnimationFrame', (_cb: FrameRequestCallback) => 0);
    vi.stubGlobal('cancelAnimationFrame', (_id: number) => undefined);
});

afterEach(() => {
    vi.unstubAllGlobals();
    document.body.innerHTML = '';
});

function makeDemo(): { demo: ForwardProcessDemo; playBtn: HTMLButtonElement; slider: HTMLInputElement } {
    const canvas = document.createElement('canvas');
    canvas.id = 'fwd-canvas';
    const slider = document.createElement('input');
    slider.type = 'range';
    slider.id = 'fwd-slider';
    const tLabel = document.createElement('span');
    tLabel.id = 'fwd-t';
    const aLabel = document.createElement('span');
    aLabel.id = 'fwd-a';
    const playBtn = document.createElement('button');
    playBtn.id = 'fwd-play';
    document.body.append(canvas, slider, tLabel, aLabel, playBtn);

    const diffusion = new DiffusionModel(); // 0-arg constructor — reads CONFIG internally
    const demo = new ForwardProcessDemo(
        'fwd-canvas', 'fwd-slider', 'fwd-t', 'fwd-a', 'fwd-play', diffusion
    );
    return { demo, playBtn, slider };
}

describe('ForwardProcessDemo — precomputation', () => {
    it('starts with zero frames before setBaseImage', () => {
        const { demo } = makeDemo();
        expect(demo.frameCount).toBe(0);
    });

    it('precomputes 401 frames (t=0..400) after setBaseImage', () => {
        const { demo } = makeDemo();
        demo.setBaseImage(new Float32Array(32 * 32).fill(0.5));
        expect(demo.frameCount).toBe(401);
    });

    it('frame[0] is identical to the base image (no noise at t=0)', () => {
        const { demo } = makeDemo();
        const img = new Float32Array(32 * 32).fill(0.5);
        demo.setBaseImage(img);
        const f0 = demo.getFrame(0)!;
        for (let i = 0; i < img.length; i++) {
            expect(f0[i]).toBeCloseTo(img[i], 5);
        }
    });

    it('frame[400] differs from frame[0] (noise was added)', () => {
        const { demo } = makeDemo();
        demo.setBaseImage(new Float32Array(32 * 32).fill(0.5));
        const f0 = demo.getFrame(0)!;
        const f400 = demo.getFrame(400)!;
        const diff = f0.reduce((sum, v, i) => sum + Math.abs(v - f400[i]), 0);
        expect(diff).toBeGreaterThan(10);
    });

    it('recomputes fresh frames when setBaseImage called again', () => {
        const { demo } = makeDemo();
        demo.setBaseImage(new Float32Array(32 * 32).fill(0.5));
        const frame1 = demo.getFrame(200)!.slice();
        demo.setBaseImage(new Float32Array(32 * 32).fill(-0.5));
        const frame2 = demo.getFrame(200)!;
        const diff = Array.from(frame1).reduce((s, v, i) => s + Math.abs(v - frame2[i]), 0);
        expect(diff).toBeGreaterThan(1);
    });
});

describe('ForwardProcessDemo — play/pause state', () => {
    it('play button shows ▶ Play initially', () => {
        const { playBtn } = makeDemo();
        expect(playBtn.textContent).toBe('▶ Play');
    });

    it('isPlaying is false initially', () => {
        const { demo } = makeDemo();
        expect(demo.isPlaying).toBe(false);
    });

    it('clicking play after setBaseImage switches button to ⏸ Pause', () => {
        const { demo, playBtn } = makeDemo();
        demo.setBaseImage(new Float32Array(32 * 32).fill(0.5));
        playBtn.click();
        expect(playBtn.textContent).toBe('⏸ Pause');
        expect(demo.isPlaying).toBe(true);
    });

    it('clicking pause switches button back to ▶ Play', () => {
        const { demo, playBtn } = makeDemo();
        demo.setBaseImage(new Float32Array(32 * 32).fill(0.5));
        playBtn.click(); // play
        playBtn.click(); // pause
        expect(playBtn.textContent).toBe('▶ Play');
        expect(demo.isPlaying).toBe(false);
    });

    it('clicking play without frames does nothing (no crash)', () => {
        const { demo, playBtn } = makeDemo();
        expect(() => playBtn.click()).not.toThrow();
        expect(demo.isPlaying).toBe(false);
    });

    it('clicking play at t=400 restarts from t=0', () => {
        const { demo, playBtn, slider } = makeDemo();
        demo.setBaseImage(new Float32Array(32 * 32).fill(0.5));
        slider.value = '400';
        slider.dispatchEvent(new Event('input'));
        expect(demo.currentT).toBe(400);
        playBtn.click();
        expect(demo.isPlaying).toBe(true);
        expect(demo.currentT).toBe(0);
    });

    it('setBaseImage while playing stops playback', () => {
        const { demo, playBtn } = makeDemo();
        demo.setBaseImage(new Float32Array(32 * 32).fill(0.5));
        playBtn.click();
        expect(demo.isPlaying).toBe(true);
        demo.setBaseImage(new Float32Array(32 * 32).fill(-0.5));
        expect(demo.isPlaying).toBe(false);
        expect(playBtn.textContent).toBe('▶ Play');
    });
});

describe('ForwardProcessDemo — slider scrub', () => {
    it('slider value changes update currentT', () => {
        const { demo, slider } = makeDemo();
        demo.setBaseImage(new Float32Array(32 * 32).fill(0.5));
        slider.value = '200';
        slider.dispatchEvent(new Event('input'));
        expect(demo.currentT).toBe(200);
    });

    it('mousedown on slider while playing pauses playback', () => {
        const { demo, playBtn, slider } = makeDemo();
        demo.setBaseImage(new Float32Array(32 * 32).fill(0.5));
        playBtn.click();
        slider.dispatchEvent(new MouseEvent('mousedown'));
        expect(demo.isPlaying).toBe(false);
    });

    it('mouseup on document resumes if was playing before scrub', () => {
        const { demo, playBtn, slider } = makeDemo();
        demo.setBaseImage(new Float32Array(32 * 32).fill(0.5));
        playBtn.click();
        slider.dispatchEvent(new MouseEvent('mousedown'));
        expect(demo.isPlaying).toBe(false);
        document.dispatchEvent(new MouseEvent('mouseup'));
        expect(demo.isPlaying).toBe(true);
    });

    it('mouseup on document stays paused if was not playing before scrub', () => {
        const { demo, slider } = makeDemo();
        demo.setBaseImage(new Float32Array(32 * 32).fill(0.5));
        slider.dispatchEvent(new MouseEvent('mousedown'));
        document.dispatchEvent(new MouseEvent('mouseup'));
        expect(demo.isPlaying).toBe(false);
    });

    it('touchstart on slider while playing pauses playback', () => {
        const { demo, playBtn, slider } = makeDemo();
        demo.setBaseImage(new Float32Array(32 * 32).fill(0.5));
        playBtn.click();
        slider.dispatchEvent(new TouchEvent('touchstart'));
        expect(demo.isPlaying).toBe(false);
    });

    it('touchend on document resumes if was playing before touch-scrub', () => {
        const { demo, playBtn, slider } = makeDemo();
        demo.setBaseImage(new Float32Array(32 * 32).fill(0.5));
        playBtn.click();
        slider.dispatchEvent(new TouchEvent('touchstart'));
        document.dispatchEvent(new TouchEvent('touchend'));
        expect(demo.isPlaying).toBe(true);
    });

    it('touchend stays paused if was not playing before touch-scrub', () => {
        const { demo, slider } = makeDemo();
        demo.setBaseImage(new Float32Array(32 * 32).fill(0.5));
        slider.dispatchEvent(new TouchEvent('touchstart'));
        document.dispatchEvent(new TouchEvent('touchend'));
        expect(demo.isPlaying).toBe(false);
    });

    it('pause-during-scrub: clicking Pause while scrubbing does not auto-resume on mouseup', () => {
        const { demo, playBtn, slider } = makeDemo();
        demo.setBaseImage(new Float32Array(32 * 32).fill(0.5));
        playBtn.click(); // start playing
        slider.dispatchEvent(new MouseEvent('mousedown')); // pauses, wasPlayingBeforeScrub=true
        playBtn.click(); // user explicitly clicks Pause → wasPlayingBeforeScrub reset
        document.dispatchEvent(new MouseEvent('mouseup')); // should NOT resume
        expect(demo.isPlaying).toBe(false);
    });

    it('setBaseImage while scrubbing does not auto-resume on mouseup', () => {
        const { demo, playBtn, slider } = makeDemo();
        demo.setBaseImage(new Float32Array(32 * 32).fill(0.5));
        playBtn.click(); // start playing
        slider.dispatchEvent(new MouseEvent('mousedown')); // wasPlayingBeforeScrub=true
        demo.setBaseImage(new Float32Array(32 * 32).fill(-0.5)); // stopPlayback() must reset flag
        document.dispatchEvent(new MouseEvent('mouseup')); // should NOT resume
        expect(demo.isPlaying).toBe(false);
    });

    it('pause-during-touch-scrub: clicking Pause while touch-scrubbing does not auto-resume on touchend', () => {
        const { demo, playBtn, slider } = makeDemo();
        demo.setBaseImage(new Float32Array(32 * 32).fill(0.5));
        playBtn.click(); // start playing
        slider.dispatchEvent(new TouchEvent('touchstart')); // pauses, wasPlayingBeforeScrub=true
        playBtn.click(); // explicit Pause → stopPlayback resets wasPlayingBeforeScrub
        document.dispatchEvent(new TouchEvent('touchend')); // should NOT resume
        expect(demo.isPlaying).toBe(false);
    });
});
