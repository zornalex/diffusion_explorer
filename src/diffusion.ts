import { CONFIG } from './config';

export class DiffusionModel {
    private T: number;
    private betas: number[];
    private alphas: number[];
    private alphasCumprod: number[];

    constructor() {
        this.T = CONFIG.timesteps;
        this.betas = this.linearBetaSchedule(CONFIG.betaStart, CONFIG.betaEnd, this.T);

        // Pre-calculate alphas and alphas_cumprod
        this.alphas = this.betas.map(beta => 1 - beta);
        this.alphasCumprod = [];

        let cumprod = 1;
        for (const alpha of this.alphas) {
            cumprod *= alpha;
            this.alphasCumprod.push(cumprod);
        }
    }

    private linearBetaSchedule(start: number, end: number, timesteps: number): number[] {
        const betas = [];
        const step = (end - start) / (timesteps - 1);
        for (let i = 0; i < timesteps; i++) {
            betas.push(start + i * step);
        }
        return betas;
    }

    // Forward process: q(x_t | x_0)
    // x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
    public addNoise(x0: Float32Array, t: number): Float32Array {
        if (t === 0) return new Float32Array(x0); // No noise at t=0

        // t is 1-indexed in papers usually, but 0-indexed here for array access
        // Let's treat input t as 0...T. 
        // If t=0, it's the clean image. 
        // If t > 0, we use index t-1 for alphasCumprod (since alphasCumprod[0] corresponds to step 1)
        // Actually, let's simplify: t goes from 0 to T.
        // t=0: Clean.
        // t=1: First step of noise.

        const alphaBar = this.alphasCumprod[t - 1];
        const sqrtAlphaBar = Math.sqrt(alphaBar);
        const sqrtOneMinusAlphaBar = Math.sqrt(1 - alphaBar);

        const noisyImage = new Float32Array(x0.length);

        for (let i = 0; i < x0.length; i++) {
            const epsilon = this.randn(); // Standard Gaussian noise
            noisyImage[i] = sqrtAlphaBar * x0[i] + sqrtOneMinusAlphaBar * epsilon;
        }

        return noisyImage;
    }

    // Box-Muller transform for standard normal distribution
    private randn(): number {
        let u = 0, v = 0;
        while (u === 0) u = Math.random();
        while (v === 0) v = Math.random();
        return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }

    public getTimesteps(): number {
        return this.T;
    }
}
