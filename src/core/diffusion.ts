import { CONFIG } from './config';

export class DiffusionModel {
    private T: number;
    private betas: number[];
    private alphas: number[];
    private alphasCumprod: number[];

    constructor() {
        this.T = CONFIG.timesteps;
        this.betas = this.cosineBetaSchedule(CONFIG.betaStart, CONFIG.betaEnd, this.T);

        // Pre-calculate alphas and alphas_cumprod
        this.alphas = this.betas.map(beta => 1 - beta);
        this.alphasCumprod = [];

        let cumprod = 1;
        for (const alpha of this.alphas) {
            cumprod *= alpha;
            this.alphasCumprod.push(cumprod);
        }
    }

    private cosineBetaSchedule(start: number, end: number, timesteps: number): number[] {
        // Cosine interpolation between betaStart and betaEnd
        const betas = [];
        for (let i = 0; i < timesteps; i++) {
            const t = i / (timesteps - 1); // 0 to 1
            // Cosine interpolation: smooth transition from start to end
            const cosine_t = (1 - Math.cos(t * Math.PI)) / 2; // 0 to 1, smooth
            const beta = start + (end - start) * cosine_t;
            betas.push(beta);
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

    // Forward process with provided epsilon (for training)
    // x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
    public addNoiseWithEpsilon(x0: Float32Array, t: number, epsilon: Float32Array): Float32Array {
        if (t === 0) return new Float32Array(x0);

        const alphaBar = this.alphasCumprod[t - 1];
        const sqrtAlphaBar = Math.sqrt(alphaBar);
        const sqrtOneMinusAlphaBar = Math.sqrt(1 - alphaBar);

        const noisyImage = new Float32Array(x0.length);

        for (let i = 0; i < x0.length; i++) {
            noisyImage[i] = sqrtAlphaBar * x0[i] + sqrtOneMinusAlphaBar * epsilon[i];
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

    public getAlpha(t: number): number {
        // t is 0-indexed (0..T-1). 
        // In paper t=1..T. 
        // Our betas array is size T. betas[0] corresponds to t=1.
        // If we pass t=0 (meaning step 1 in paper), we want betas[0].
        if (t < 0 || t >= this.T) return 1;
        return this.alphas[t];
    }

    public getAlphaBar(t: number): number {
        if (t < 0) return 1; // alpha_bar_0 = 1
        if (t >= this.T) return 0;
        return this.alphasCumprod[t];
    }
}
