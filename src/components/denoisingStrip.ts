import type { DenoisingFrame } from '../core/trainer';
import { CONFIG } from '../core/config';

const THUMB_SIZE = 48; // px display size for thumbnails
const INTERNAL   = CONFIG.imageSize;

/**
 * Renders the denoising strip (row of thumbnails) and manages
 * the zoom-in panel showing [xₜ] → [ε̂] → [xₜ₋₁].
 */
export class DenoisingStrip {
    private container: HTMLElement;
    private zoomPanel: HTMLElement;
    private frames: DenoisingFrame[] = [];
    private activeIdx = -1;

    constructor(containerId: string, zoomPanelId: string) {
        this.container = document.getElementById(containerId)!;
        this.zoomPanel = document.getElementById(zoomPanelId)!;
        this.setupZoomClose();
    }

    /** Load frames from trainer and render the strip */
    public load(frames: DenoisingFrame[]) {
        this.frames = frames;
        this.render();
    }

    private render() {
        this.container.innerHTML = '';

        this.frames.forEach((frame, idx) => {
            const wrap = document.createElement('div');
            wrap.className = 'strip-thumb';
            wrap.title = `t = ${frame.t}`;

            const canvas = document.createElement('canvas');
            canvas.width  = INTERNAL;
            canvas.height = INTERNAL;
            canvas.style.width  = `${THUMB_SIZE}px`;
            canvas.style.height = `${THUMB_SIZE}px`;
            canvas.style.imageRendering = 'pixelated';
            this.renderFloat32(frame.xt, canvas);

            const label = document.createElement('span');
            label.className = 'strip-label';
            label.textContent = `t=${frame.t}`;

            wrap.appendChild(canvas);
            wrap.appendChild(label);
            wrap.addEventListener('click', () => this.openZoom(idx));
            this.container.appendChild(wrap);
        });

        // Scroll to first frame (pure noise on left)
        this.container.scrollLeft = 0;
    }

    private openZoom(idx: number) {
        const frame = this.frames[idx];
        this.activeIdx = idx;

        // Mark active thumbnail
        this.container.querySelectorAll('.strip-thumb').forEach((el, i) => {
            el.classList.toggle('active', i === idx);
        });

        // Populate zoom panel
        this.renderZoomCanvas('zoom-xt',     frame.xt);
        this.renderZoomCanvas('zoom-eps',    frame.epsPred);
        this.renderZoomCanvas('zoom-xprev',  frame.xPrev);

        const tLabel = this.zoomPanel.querySelector('.zoom-t-label');
        if (tLabel) tLabel.textContent = `t = ${frame.t}`;

        const prevBtn = this.zoomPanel.querySelector('.zoom-nav-prev') as HTMLButtonElement;
        const nextBtn = this.zoomPanel.querySelector('.zoom-nav-next') as HTMLButtonElement;
        if (prevBtn) prevBtn.disabled = idx >= this.frames.length - 1;
        if (nextBtn) nextBtn.disabled = idx <= 0;

        this.zoomPanel.classList.add('visible');
    }

    private renderZoomCanvas(id: string, data: Float32Array) {
        const canvas = document.getElementById(id) as HTMLCanvasElement;
        if (!canvas) return;
        canvas.width  = INTERNAL;
        canvas.height = INTERNAL;
        this.renderFloat32(data, canvas);
    }

    private renderFloat32(data: Float32Array, canvas: HTMLCanvasElement) {
        const ctx = canvas.getContext('2d')!;
        const img = ctx.createImageData(INTERNAL, INTERNAL);
        for (let i = 0; i < data.length; i++) {
            const v = Math.max(0, Math.min(255, (data[i] + 1) * 127.5));
            img.data[i * 4]     = v;
            img.data[i * 4 + 1] = v;
            img.data[i * 4 + 2] = v;
            img.data[i * 4 + 3] = 255;
        }
        ctx.putImageData(img, 0, 0);
    }

    private setupZoomClose() {
        // Close button
        const closeBtn = this.zoomPanel.querySelector('.zoom-close');
        closeBtn?.addEventListener('click', () => {
            this.zoomPanel.classList.remove('visible');
            this.container.querySelectorAll('.strip-thumb').forEach(el => el.classList.remove('active'));
        });

        // Navigation (prev = higher t, next = lower t)
        this.zoomPanel.querySelector('.zoom-nav-prev')?.addEventListener('click', () => {
            if (this.activeIdx < this.frames.length - 1) this.openZoom(this.activeIdx + 1);
        });
        this.zoomPanel.querySelector('.zoom-nav-next')?.addEventListener('click', () => {
            if (this.activeIdx > 0) this.openZoom(this.activeIdx - 1);
        });
    }

    public clear() {
        this.frames = [];
        this.container.innerHTML = '<span class="strip-empty">Generate to see denoising steps</span>';
        this.zoomPanel.classList.remove('visible');
    }
}
