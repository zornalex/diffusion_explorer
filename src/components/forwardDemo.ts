import type { DiffusionModel } from '../core/diffusion';
import { renderImageToCanvas, setupCanvas } from './renderUtils';

const FRAME_INTERVAL_MS = 15; // ~6 sec for 400 steps

/**
 * Interactive Forward Process Demo.
 *
 * On setBaseImage(): precomputes all 401 frames (t=0..400) using the
 * closed-form noise formula with a fixed epsilon, caches as Float32Array[].
 *
 * Play button: animates through cached frames at ~15ms/frame.
 * Slider: always scrubable; dragging pauses animation and resumes on release
 * if the animation was playing.
 */
export class ForwardProcessDemo {
    private readonly ctx: CanvasRenderingContext2D;
    private readonly slider: HTMLInputElement;
    private readonly tLabel: HTMLElement;
    private readonly alphaBarLabel: HTMLElement;
    private readonly playBtn: HTMLButtonElement;
    private readonly diffusion: DiffusionModel;

    private baseImage: Float32Array | null = null;
    private epsilon: Float32Array | null = null;
    private frames: Float32Array[] = [];

    private _isPlaying = false;
    private _currentT = 0;
    private rafId: number | null = null;
    private lastFrameTime = 0;
    private wasPlayingBeforeScrub = false;
    // Stored so destroy() can removeEventListener; assigned in bindSlider() via constructor
    private onDocMouseUp!: () => void;
    private onDocTouchEnd!: () => void;

    // ── Public read-only accessors (used by tests) ──────────────────────────

    get frameCount(): number { return this.frames.length; }
    get isPlaying(): boolean { return this._isPlaying; }
    get currentT(): number { return this._currentT; }

    public getFrame(t: number): Float32Array | undefined {
        return this.frames[t];
    }

    // ── Constructor ─────────────────────────────────────────────────────────

    constructor(
        canvasId: string,
        sliderId: string,
        tLabelId: string,
        alphaBarLabelId: string,
        playBtnId: string,
        diffusion: DiffusionModel
    ) {
        this.diffusion = diffusion;
        this.ctx = setupCanvas(canvasId, 32, '200px');
        this.slider = document.getElementById(sliderId) as HTMLInputElement;
        this.tLabel = document.getElementById(tLabelId) as HTMLElement;
        this.alphaBarLabel = document.getElementById(alphaBarLabelId) as HTMLElement;
        this.playBtn = document.getElementById(playBtnId) as HTMLButtonElement;

        const T = diffusion.getTimesteps();
        this.slider.min = '0';
        this.slider.max = T.toString();
        this.slider.value = '0';

        this.bindSlider();
        this.bindPlayBtn();
        this.updatePlayBtn();
        this.renderPlaceholder();
    }

    // ── Public API ──────────────────────────────────────────────────────────

    /** Load a new base image, precompute all frames, reset to t=0. */
    public setBaseImage(img: Float32Array): void {
        this.stopPlayback();
        this.baseImage = img;
        this.epsilon = this.sampleGaussian(img.length);
        this.precompute();
        this._currentT = 0;
        this.slider.value = '0';
        this.renderFromCache(0);
    }

    /** Release rAF and document-level listeners. Call when removing the component from the DOM. */
    public destroy(): void {
        this.stopPlayback();
        document.removeEventListener('mouseup', this.onDocMouseUp);
        document.removeEventListener('touchend', this.onDocTouchEnd);
    }

    // ── Precomputation ──────────────────────────────────────────────────────

    private precompute(): void {
        if (!this.baseImage || !this.epsilon) return;
        const T = this.diffusion.getTimesteps();
        this.frames = new Array(T + 1);
        this.frames[0] = new Float32Array(this.baseImage); // clean copy at t=0
        for (let t = 1; t <= T; t++) {
            this.frames[t] = this.diffusion.addNoiseWithEpsilon(this.baseImage, t, this.epsilon);
        }
    }

    // ── Play / Pause ─────────────────────────────────────────────────────────

    private bindPlayBtn(): void {
        this.playBtn.addEventListener('click', () => {
            if (this.frames.length === 0) return; // no image loaded yet
            // If mid-scrub (wasPlayingBeforeScrub=true), user explicitly cancels auto-resume
            if (this.wasPlayingBeforeScrub) {
                this.wasPlayingBeforeScrub = false;
                return; // do not toggle play state — scrub pause stays paused
            }
            if (this._isPlaying) {
                this.stopPlayback();
            } else {
                if (this._currentT >= this.diffusion.getTimesteps()) {
                    this._currentT = 0; // restart from beginning if at end
                    this.slider.value = '0';
                }
                this.startPlayback();
            }
        });
    }

    private startPlayback(): void {
        this._isPlaying = true;
        this.lastFrameTime = 0;
        this.updatePlayBtn();
        this.scheduleFrame();
    }

    private stopPlayback(): void {
        this._isPlaying = false;
        this.wasPlayingBeforeScrub = false; // prevent stale resume after setBaseImage or any other direct caller
        if (this.rafId !== null) {
            cancelAnimationFrame(this.rafId);
            this.rafId = null;
        }
        this.updatePlayBtn();
    }

    private scheduleFrame(): void {
        const step = (now: number) => {
            if (!this._isPlaying) return;

            if (now - this.lastFrameTime >= FRAME_INTERVAL_MS) {
                this.lastFrameTime = now;
                this._currentT++;

                const T = this.diffusion.getTimesteps();
                if (this._currentT >= T) {
                    this._currentT = T;
                    this.slider.value = String(T);
                    this.renderFromCache(T);
                    this.stopPlayback(); // animation finished
                    return;
                }

                this.slider.value = String(this._currentT);
                this.renderFromCache(this._currentT);
            }

            this.rafId = requestAnimationFrame(step);
        };

        this.rafId = requestAnimationFrame(step);
    }

    private updatePlayBtn(): void {
        this.playBtn.textContent = this._isPlaying ? '⏸ Pause' : '▶ Play';
    }

    // ── Slider scrubbing ─────────────────────────────────────────────────────

    private bindSlider(): void {
        // Live render as slider moves
        this.slider.addEventListener('input', () => {
            const t = parseInt(this.slider.value, 10);
            this._currentT = t;
            this.renderFromCache(t);
        });

        // Pause on drag start, resume on drag end if was playing
        this.slider.addEventListener('mousedown', () => {
            const wasPlaying = this._isPlaying;
            if (this._isPlaying) this.stopPlayback(); // resets wasPlayingBeforeScrub to false
            this.wasPlayingBeforeScrub = wasPlaying;  // restore intent after stopPlayback
        });
        this.slider.addEventListener('touchstart', () => {
            const wasPlaying = this._isPlaying;
            if (this._isPlaying) this.stopPlayback(); // resets wasPlayingBeforeScrub to false
            this.wasPlayingBeforeScrub = wasPlaying;  // restore intent after stopPlayback
        }, { passive: true });

        this.onDocMouseUp = () => {
            if (this.wasPlayingBeforeScrub) {
                this.wasPlayingBeforeScrub = false;
                this.startPlayback();
            }
        };
        this.onDocTouchEnd = () => {
            if (this.wasPlayingBeforeScrub) {
                this.wasPlayingBeforeScrub = false;
                this.startPlayback();
            }
        };
        document.addEventListener('mouseup', this.onDocMouseUp);
        document.addEventListener('touchend', this.onDocTouchEnd);
    }

    // ── Rendering ────────────────────────────────────────────────────────────

    private renderFromCache(t: number): void {
        if (this.frames.length === 0) return;
        const frame = this.frames[t];
        if (!frame) return;
        renderImageToCanvas(frame, this.ctx);
        this.updateLabels(t);
    }

    private updateLabels(t: number): void {
        const T = this.diffusion.getTimesteps();
        this.tLabel.textContent = `t = ${t}`;
        if (t === 0) {
            this.alphaBarLabel.textContent = 'ᾱ_t = 1.0000';
        } else if (t >= T) {
            this.alphaBarLabel.textContent = 'ᾱ_t ≈ 0.0000';
        } else {
            // getAlphaBar uses 0-index: getAlphaBar(t-1) = alphasCumprod[t-1]
            // matches addNoiseWithEpsilon which also reads alphasCumprod[t-1] for step t
            const abar = this.diffusion.getAlphaBar(t - 1);
            this.alphaBarLabel.textContent = `ᾱ_t = ${abar.toFixed(4)}`;
        }
    }

    private renderPlaceholder(): void {
        const { canvas } = this.ctx;
        this.ctx.fillStyle = '#1a1a1a';
        this.ctx.fillRect(0, 0, canvas.width, canvas.height);
        this.ctx.fillStyle = '#334155';
        this.ctx.font = `${Math.floor(canvas.width / 6)}px sans-serif`; // ~1/6 of canvas width for the '?' placeholder glyph
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        this.ctx.fillText('?', canvas.width / 2, canvas.height / 2);
        this.ctx.textAlign = 'left';
        this.ctx.textBaseline = 'alphabetic';
    }

    /** Box-Muller: sample n i.i.d. standard-normal values.
     *  Note: DiffusionModel has a private randn() with identical logic.
     *  Duplicated here because randn() is private and not tensor-based — extracting
     *  to a shared util is reasonable if a third caller appears. */
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
}
