import * as tf from '@tensorflow/tfjs';
import { CONFIG } from './config';
import { DiffusionModel } from './diffusion';

export interface DebugFrame {
    t: number;
    x0: Float32Array;
    epsilon: Float32Array;
    xt: Float32Array;
}

/** One cached denoising frame: the image at timestep t, plus what the model predicted */
export interface DenoisingFrame {
    t: number;
    xt: Float32Array;          // noisy image at this step
    epsPred: Float32Array;     // predicted noise by the model
    xPrev: Float32Array;       // result after one denoising step
}

export class ModelTrainer {
    private model: tf.LayersModel;
    private diffusion: DiffusionModel;
    private isTraining: boolean = false;

    /** All denoising frames from the last generateImage() call */
    public lastFrames: DenoisingFrame[] = [];

    constructor() {
        this.diffusion = new DiffusionModel();
        this.model = this.createModel();
    }

    // ─── Model init ─────────────────────────────────────────────────────────

    public async init() {
        try {
            const model = await tf.loadLayersModel('/model/model.json');
            model.compile({ optimizer: tf.train.adam(CONFIG.learningRate), loss: 'meanSquaredError' });
            this.model.dispose();
            this.model = model;
            console.log('✅ Pretrained model loaded');
        } catch (err) {
            console.warn('Pretrained model load failed:', err);
            console.log('No pretrained model found, using fresh model');
        }
    }

    // ─── Sinusoidal time embedding ────────────────────────────────────────────
    // Maps t_norm ∈ [0,1] → dim-dimensional vector of sin/cos features at
    // exponentially-spaced frequencies.  Unlike a raw scalar, every timestep
    // produces a unique signature that the Dense layers can immediately use —
    // no need to "learn how to represent time" from scratch.
    private sinEmb(tNorm: number, dim: number): Float32Array {
        const out    = new Float32Array(dim);
        const scaled = tNorm * 1000; // stretch [0,1] → [0,1000] for multi-scale coverage
        for (let i = 0; i < dim / 2; i++) {
            const freq     = 1.0 / Math.pow(10000, (2 * i) / dim);
            out[2 * i]     = Math.sin(scaled * freq);
            out[2 * i + 1] = Math.cos(scaled * freq);
        }
        return out;
    }

    // ─── Architecture ────────────────────────────────────────────────────────
    // U-Net with two time-conditioning points:
    //   1. INPUT : sin-emb → Dense(TD→S²) → 32×32 spatial map, concat with image
    //   2. BOTTLENECK : sin-emb → Dense(TD→64) → add to 8×8×64 features
    // This keeps the time signal alive at both fine (pixel) and coarse (semantic)
    // scales, which is the key insight from modern diffusion U-Net designs.

    private createModel(): tf.LayersModel {
        const S  = CONFIG.imageSize;   // 32
        const TD = CONFIG.timeDim;     // 16

        const inputImage = tf.input({ shape: [S, S, 1], name: 'input_image' });
        const inputTime  = tf.input({ shape: [TD],       name: 'input_time'  });

        // ── 1. Input-level time conditioning ─────────────────────────────────
        // Linear projection (no relu — sin/cos features span [-1,1], relu kills half)
        const timeProj = tf.layers.dense({ units: S * S, name: 'time_dense' })
            .apply(inputTime) as tf.SymbolicTensor;                          // [B, S²]
        const timeMap  = tf.layers.reshape({ targetShape: [S, S, 1], name: 'time_reshape' })
            .apply(timeProj) as tf.SymbolicTensor;                           // [B, S, S, 1]

        const cat0 = tf.layers.concatenate({ name: 'cat_input' })
            .apply([inputImage, timeMap]) as tf.SymbolicTensor;              // [B, S, S, 2]

        // ── Encoder ──────────────────────────────────────────────────────────
        const e1  = tf.layers.conv2d({ filters: 16, kernelSize: 3, padding: 'same', activation: 'relu', kernelInitializer: 'heNormal', name: 'enc1a' }).apply(cat0) as tf.SymbolicTensor;
        const e1b = tf.layers.conv2d({ filters: 16, kernelSize: 3, padding: 'same', activation: 'relu', kernelInitializer: 'heNormal', name: 'enc1b' }).apply(e1)   as tf.SymbolicTensor;
        const p1  = tf.layers.maxPooling2d({ poolSize: [2, 2], name: 'pool1' }).apply(e1b) as tf.SymbolicTensor; // 16×16

        const e2  = tf.layers.conv2d({ filters: 32, kernelSize: 3, padding: 'same', activation: 'relu', kernelInitializer: 'heNormal', name: 'enc2a' }).apply(p1)  as tf.SymbolicTensor;
        const e2b = tf.layers.conv2d({ filters: 32, kernelSize: 3, padding: 'same', activation: 'relu', kernelInitializer: 'heNormal', name: 'enc2b' }).apply(e2)  as tf.SymbolicTensor;
        const p2  = tf.layers.maxPooling2d({ poolSize: [2, 2], name: 'pool2' }).apply(e2b) as tf.SymbolicTensor; // 8×8

        // ── Bottleneck ────────────────────────────────────────────────────────
        const b  = tf.layers.conv2d({ filters: 64, kernelSize: 3, padding: 'same', activation: 'relu', kernelInitializer: 'heNormal', name: 'bot_a' }).apply(p2) as tf.SymbolicTensor;
        const bb = tf.layers.conv2d({ filters: 64, kernelSize: 3, padding: 'same', activation: 'relu', kernelInitializer: 'heNormal', name: 'bot_b' }).apply(b)  as tf.SymbolicTensor;

        // ── 2. Bottleneck-level time conditioning ─────────────────────────────
        // Project sin-emb → 64-dim bias, reshape to [1,1,64], broadcast-add to
        // the [B,8,8,64] bottleneck so high-level features stay time-aware.
        const timeBotProj = tf.layers.dense({ units: 64, activation: 'relu', name: 'time_bot' })
            .apply(inputTime) as tf.SymbolicTensor;                          // [B, 64]
        const timeBotMap  = tf.layers.reshape({ targetShape: [1, 1, 64], name: 'time_bot_reshape' })
            .apply(timeBotProj) as tf.SymbolicTensor;                        // [B, 1, 1, 64]
        const botCond = tf.layers.add({ name: 'bot_time_add' })
            .apply([bb, timeBotMap]) as tf.SymbolicTensor;                   // [B, 8, 8, 64]

        // ── Decoder with skip connections ─────────────────────────────────────
        const u2   = tf.layers.upSampling2d({ size: [2, 2], name: 'up2' }).apply(botCond)     as tf.SymbolicTensor; // 16×16
        const cat2 = tf.layers.concatenate({ name: 'cat2' }).apply([u2, e2b])                 as tf.SymbolicTensor; // 16×16×96
        const d2   = tf.layers.conv2d({ filters: 32, kernelSize: 3, padding: 'same', activation: 'relu', kernelInitializer: 'heNormal', name: 'dec2a' }).apply(cat2) as tf.SymbolicTensor;
        const d2b  = tf.layers.conv2d({ filters: 32, kernelSize: 3, padding: 'same', activation: 'relu', kernelInitializer: 'heNormal', name: 'dec2b' }).apply(d2)   as tf.SymbolicTensor;

        const u1   = tf.layers.upSampling2d({ size: [2, 2], name: 'up1' }).apply(d2b)         as tf.SymbolicTensor; // 32×32
        const cat1 = tf.layers.concatenate({ name: 'cat1' }).apply([u1, e1b])                 as tf.SymbolicTensor; // 32×32×48
        const d1   = tf.layers.conv2d({ filters: 16, kernelSize: 3, padding: 'same', activation: 'relu', kernelInitializer: 'heNormal', name: 'dec1' }).apply(cat1) as tf.SymbolicTensor;

        // ── Output ────────────────────────────────────────────────────────────
        const output = tf.layers.conv2d({ filters: 1, kernelSize: 1, padding: 'same', kernelInitializer: 'zeros', name: 'output' }).apply(d1) as tf.SymbolicTensor;

        const model = tf.model({ inputs: [inputImage, inputTime], outputs: output });
        model.compile({ optimizer: tf.train.adam(CONFIG.learningRate), loss: 'meanSquaredError' });
        return model;
    }

    // ─── Training ────────────────────────────────────────────────────────────

    /**
     * Train on a dataset of multiple images (e.g. several smileys).
     * At each training step, a random image from the dataset is picked.
     */
    public async trainOnDataset(
        images: Float32Array[],
        onStepEnd: (step: number, loss: number, debug?: DebugFrame) => void
    ) {
        if (this.isTraining || images.length === 0) return;
        this.isTraining = true;

        const batchSize = CONFIG.batchSize;
        const T  = this.diffusion.getTimesteps();
        const S  = CONFIG.imageSize;

        for (let step = 0; step < CONFIG.trainingSteps; step++) {
            if (!this.isTraining) break;

            // ── CPU: pick random timesteps + images ───────────────────────────
            const tVals = Array.from({ length: batchSize }, () =>
                Math.floor(Math.random() * T) + 1
            );
            // Build image batch as a flat Float32Array → single tensor upload
            const imgFlat = new Float32Array(batchSize * S * S);
            for (let i = 0; i < batchSize; i++) {
                const img = images[Math.floor(Math.random() * images.length)];
                imgFlat.set(img, i * S * S);
            }

            // ── GPU / TF-backend: keep everything as tensors, no round-trips ─
            const x0Batch  = tf.tensor4d(imgFlat, [batchSize, S, S, 1]);
            const epsBatch = tf.randomNormal([batchSize, S, S, 1]) as tf.Tensor4D;

            // Pre-computed ᾱ coefficients
            const alphaBars  = tVals.map(t => this.diffusion.getAlphaBar(t - 1));
            const sqrtAB     = tf.tensor(alphaBars.map(Math.sqrt)).reshape([batchSize, 1, 1, 1]) as tf.Tensor4D;
            const sqrtOMAB   = tf.tensor(alphaBars.map(a => Math.sqrt(1 - a))).reshape([batchSize, 1, 1, 1]) as tf.Tensor4D;

            // x_t = √ᾱ_t · x₀ + √(1−ᾱ_t) · ε
            const xtBatch = x0Batch.mul(sqrtAB).add(epsBatch.mul(sqrtOMAB)) as tf.Tensor4D;

            // ── Sinusoidal time embedding [batchSize, TD] ─────────────────────
            const TD      = CONFIG.timeDim;
            const tEmbFlat = new Float32Array(batchSize * TD);
            tVals.forEach((t, i) => tEmbFlat.set(this.sinEmb(t / T, TD), i * TD));
            const tBatch  = tf.tensor2d(tEmbFlat, [batchSize, TD]);

            const history = await this.model.fit([xtBatch, tBatch], epsBatch, { epochs: 1, verbose: 0 });
            const loss    = history.history.loss[0] as number;

            if (isNaN(loss)) {
                tf.dispose([x0Batch, epsBatch, sqrtAB, sqrtOMAB, xtBatch, tBatch]);
                this.isTraining = false;
                break;
            }

            // ── Debug frame: download only one item, only every N steps ──────
            let debug: DebugFrame | undefined;
            if (step % CONFIG.debugFrameInterval === 0) {
                const [xtData, epsData] = await Promise.all([
                    xtBatch.slice([0, 0, 0, 0], [1, S, S, 1]).data(),
                    epsBatch.slice([0, 0, 0, 0], [1, S, S, 1]).data(),
                ]);
                debug = {
                    t:       tVals[0],
                    x0:      images[0],
                    epsilon: new Float32Array(epsData),
                    xt:      new Float32Array(xtData),
                };
            }

            tf.dispose([x0Batch, epsBatch, sqrtAB, sqrtOMAB, xtBatch, tBatch]);
            onStepEnd(step + 1, loss, debug);
            if (step % CONFIG.nextFrameInterval === 0) await tf.nextFrame();
        }

        this.isTraining = false;
    }

    /** Legacy single-image training (kept for compatibility) */
    public async trainOnSingleImage(
        imageData: Float32Array,
        onStepEnd: (step: number, loss: number, debug?: DebugFrame) => void
    ) {
        return this.trainOnDataset([imageData], onStepEnd);
    }

    public stopTraining() { this.isTraining = false; }

    // ─── Generation with frame caching ──────────────────────────────────────

    /**
     * Run DDPM reverse process. Caches all frames into this.lastFrames.
     * Calls onStep every `reportEvery` steps for live UI update.
     */
    public async generateImage(
        onStep: (t: number, imageData: Float32Array, progress: number) => void,
        reportEvery = 1
    ) {
        this.lastFrames = [];
        let x = tf.randomNormal([1, CONFIG.imageSize, CONFIG.imageSize, 1]);
        const T = this.diffusion.getTimesteps();
        const eps = 1e-8;

        const TD = CONFIG.timeDim;

        for (let t = T; t >= 1; t--) {
            const tNorm   = t / T;
            const tTensor = tf.tensor2d([Array.from(this.sinEmb(tNorm, TD))], [1, TD]);

            const predictedNoise = this.model.predict([x, tTensor]) as tf.Tensor;

            const alpha_t        = this.diffusion.getAlpha(t - 1);
            const alphaBar_t     = this.diffusion.getAlphaBar(t - 1);
            const alphaBar_prev  = this.diffusion.getAlphaBar(t - 2);
            const beta_t         = 1 - alpha_t;
            const sqrtAlpha      = Math.sqrt(alpha_t);
            const sqrtOMaB       = Math.sqrt(Math.max(1 - alphaBar_t, eps));
            const betaTilde      = (Math.max(1 - alphaBar_prev, eps) / Math.max(1 - alphaBar_t, eps)) * beta_t;
            const sigma_t        = Math.sqrt(Math.max(betaTilde, 0));
            const coeff          = beta_t / sqrtOMaB;

            const clippedNoise   = tf.clipByValue(predictedNoise, -1.5, 1.5);
            const z              = t > 1 ? tf.randomNormal([1, CONFIG.imageSize, CONFIG.imageSize, 1]) : tf.zeros([1, CONFIG.imageSize, CONFIG.imageSize, 1]);
            const xPrevRaw       = x.sub(clippedNoise.mul(coeff)).div(sqrtAlpha).add(z.mul(sigma_t));
            const xPrev          = tf.clipByValue(xPrevRaw, -1.2, 1.2);

            // Cache frame (every ~20 steps for strip; always store t=0)
            if (t % 20 === 0 || t === 1 || t === T) {
                const [xtData, epsData, xpData] = await Promise.all([
                    x.data(),
                    clippedNoise.data(),
                    xPrev.data(),
                ]);
                this.lastFrames.push({
                    t,
                    xt:      new Float32Array(xtData),
                    epsPred: new Float32Array(epsData),
                    xPrev:   new Float32Array(xpData),
                });
            }

            x.dispose();
            xPrevRaw.dispose();
            x = xPrev;
            tf.dispose([tTensor, predictedNoise, clippedNoise, z]);

            if (t % reportEvery === 0 || t === 1) {
                const data = await x.data() as Float32Array;
                onStep(t, data, T - t);
            }

            await tf.nextFrame();
        }

        x.dispose();

        // Sort frames t descending (T → 0) for strip display
        this.lastFrames.sort((a, b) => b.t - a.t);
    }

    // ─── Save / Load ─────────────────────────────────────────────────────────

    public async saveModel(name = 'diffusion-model') {
        try {
            await this.model.save(`indexeddb://${name}`);
            await this.model.save(`downloads://${name}`);
            return true;
        } catch (e) {
            console.error('Save failed:', e);
            return false;
        }
    }

    public async loadModelFromIndexedDB(name = 'diffusion-model') {
        try {
            const m = await tf.loadLayersModel(`indexeddb://${name}`);
            m.compile({ optimizer: tf.train.adam(CONFIG.learningRate), loss: 'meanSquaredError' });
            this.model.dispose();
            this.model = m;
            return true;
        } catch { return false; }
    }

    public async loadModelFromFiles(files: FileList) {
        try {
            const m = await tf.loadLayersModel(tf.io.browserFiles(Array.from(files)));
            m.compile({ optimizer: tf.train.adam(CONFIG.learningRate), loss: 'meanSquaredError' });
            this.model.dispose();
            this.model = m;
            return true;
        } catch { return false; }
    }
}
