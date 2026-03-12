import type { DiffusionModel } from '../core/diffusion';
import { renderImageToCanvas, setupCanvas } from './renderUtils';

/**
 * Interactive Forward Process Demo.
 * Slider t=0..T controls noise level on a preset smiley image.
 * Calls DiffusionModel.addNoise() and re-renders canvas on every slider move.
 */
export class ForwardProcessDemo {
    private ctx: CanvasRenderingContext2D;
    private slider: HTMLInputElement;
    private tLabel: HTMLElement;
    private alphaBarLabel: HTMLElement;
    private diffusion: DiffusionModel;
    private baseImage: Float32Array | null = null;
    /** Fixed noise sample — regenerated whenever a new base image is loaded.
     *  Ensures the slider shows a consistent, monotonically noisier trajectory
     *  (t=70 looks like t=50 + more noise, not a completely different realization). */
    private epsilon: Float32Array | null = null;

    constructor(
        canvasId: string,
        sliderId: string,
        tLabelId: string,
        alphaBarLabelId: string,
        diffusion: DiffusionModel
    ) {
        this.diffusion = diffusion;
        this.ctx = setupCanvas(canvasId, 32, '200px');
        this.slider = document.getElementById(sliderId) as HTMLInputElement;
        this.tLabel = document.getElementById(tLabelId) as HTMLElement;
        this.alphaBarLabel = document.getElementById(alphaBarLabelId) as HTMLElement;

        this.slider.min = '0';
        this.slider.max = diffusion.getTimesteps().toString();
        this.slider.value = '0';

        this.slider.addEventListener('input', () => {
            this.render(parseInt(this.slider.value));
        });

        this.renderPlaceholder();
    }

    /** Load a new base image and reset slider to t=0.
     *  A fresh fixed epsilon is sampled so all subsequent slider positions
     *  use the same noise realization → t=70 visually "builds on" t=50. */
    public setBaseImage(img: Float32Array) {
        this.baseImage = img;
        this.epsilon = this.sampleGaussian(img.length);
        this.slider.value = '0';
        this.render(0);
    }

    /** Box-Muller: sample n i.i.d. standard-normal values */
    private sampleGaussian(n: number): Float32Array {
        const out = new Float32Array(n);
        for (let i = 0; i < n; i++) {
            let u = 0, v = 0;
            while (u === 0) u = Math.random();
            while (v === 0) v = Math.random();
            out[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
        }
        return out;
    }

    private render(t: number) {
        if (!this.baseImage || !this.epsilon) return;

        const T = this.diffusion.getTimesteps();
        const noisy = t === 0
            ? this.baseImage
            : this.diffusion.addNoiseWithEpsilon(this.baseImage, t, this.epsilon);
        renderImageToCanvas(noisy, this.ctx);

        // Update labels
        this.tLabel.textContent = `t = ${t}`;
        if (t === 0) {
            this.alphaBarLabel.textContent = 'ᾱ_t = 1.0000';
        } else if (t >= T) {
            this.alphaBarLabel.textContent = 'ᾱ_t ≈ 0.0000';
        } else {
            const abar = this.diffusion.getAlphaBar(t - 1);
            this.alphaBarLabel.textContent = `ᾱ_t = ${abar.toFixed(4)}`;
        }
    }

    private renderPlaceholder() {
        const ctx = this.ctx;
        const canvas = ctx.canvas;
        ctx.fillStyle = '#1a1a1a';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#334155';
        ctx.font = `${Math.floor(canvas.width / 6)}px sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('?', canvas.width / 2, canvas.height / 2);
        ctx.textAlign = 'left';
        ctx.textBaseline = 'alphabetic';
    }
}
