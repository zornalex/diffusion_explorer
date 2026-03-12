import { CONFIG } from '../core/config';

const DISPLAY_SIZE = 80; // px
const INTERNAL = CONFIG.imageSize;

/**
 * Manages the multi-image drawing panel (Teachable Machine style).
 * Each slot: mini DrawingCanvas + clear button.
 */
export class DatasetPanel {
    private slots: SlotCanvas[] = [];
    private containerId: string;

    constructor(containerId: string, numSlots = 5) {
        this.containerId = containerId;
        this.init(numSlots);
    }

    private init(n: number) {
        const container = document.getElementById(this.containerId)!;
        container.innerHTML = '';
        for (let i = 0; i < n; i++) {
            const slot = new SlotCanvas(i, DISPLAY_SIZE, INTERNAL);
            container.appendChild(slot.element);
            this.slots.push(slot);
        }
    }

    /** Return all non-empty slot images as Float32Arrays */
    public getImages(): Float32Array[] {
        return this.slots
            .filter(s => s.hasContent)
            .map(s => s.getFloat32());
    }

    /** Load preset images into slots */
    public loadPresets(images: Float32Array[]) {
        images.forEach((img, i) => {
            if (i < this.slots.length) {
                this.slots[i].setFloat32(img);
            }
        });
    }

    /** Clear all slots */
    public clearAll() {
        this.slots.forEach(s => s.clear());
    }

    public get count() {
        return this.slots.filter(s => s.hasContent).length;
    }
}

class SlotCanvas {
    public element: HTMLElement;
    public hasContent = false;

    private canvas: HTMLCanvasElement;
    private ctx: CanvasRenderingContext2D;
    private isDrawing = false;
    private internal: number;

    constructor(idx: number, displaySize: number, internal: number) {
        this.internal = internal;

        this.element = document.createElement('div');
        this.element.className = 'dataset-slot';

        this.canvas = document.createElement('canvas');
        this.canvas.width  = internal;
        this.canvas.height = internal;
        this.canvas.style.width  = `${displaySize}px`;
        this.canvas.style.height = `${displaySize}px`;
        this.canvas.style.imageRendering = 'pixelated';
        this.canvas.style.cursor = 'crosshair';
        this.canvas.title = `Slot ${idx + 1}`;

        this.ctx = this.canvas.getContext('2d', { willReadFrequently: true })!;
        this.fill(-1); // black background

        const clearBtn = document.createElement('button');
        clearBtn.className = 'slot-clear-btn btn-icon';
        clearBtn.title = 'Clear';
        clearBtn.textContent = '×';
        clearBtn.addEventListener('click', (e) => { e.stopPropagation(); this.clear(); });

        this.element.appendChild(this.canvas);
        this.element.appendChild(clearBtn);

        this.initDrawing();
    }

    private getPos(e: MouseEvent | Touch): { x: number; y: number } {
        const rect = this.canvas.getBoundingClientRect();
        return {
            x: ((e.clientX - rect.left) / rect.width) * this.internal,
            y: ((e.clientY - rect.top) / rect.height) * this.internal,
        };
    }

    private initDrawing() {
        this.canvas.addEventListener('mousedown', (e) => { this.isDrawing = true; this.drawAt(e); });
        this.canvas.addEventListener('mousemove', (e) => { if (this.isDrawing) this.drawAt(e); });
        this.canvas.addEventListener('mouseup',   () => { this.isDrawing = false; });
        this.canvas.addEventListener('mouseleave',() => { this.isDrawing = false; });

        this.canvas.addEventListener('touchstart', (e) => { e.preventDefault(); this.isDrawing = true; this.drawAt(e.touches[0]); });
        this.canvas.addEventListener('touchmove',  (e) => { e.preventDefault(); if (this.isDrawing) this.drawAt(e.touches[0]); });
        this.canvas.addEventListener('touchend',   () => { this.isDrawing = false; });
    }

    private drawAt(e: MouseEvent | Touch) {
        const { x, y } = this.getPos(e);
        this.ctx.fillStyle = 'white';
        this.ctx.beginPath();
        this.ctx.arc(x, y, 2, 0, Math.PI * 2);
        this.ctx.fill();
        this.hasContent = true;
    }

    public clear() {
        this.fill(-1);
        this.hasContent = false;
    }

    private fill(value: number) {
        const col = value >= 0 ? 'white' : 'black';
        this.ctx.fillStyle = col;
        this.ctx.fillRect(0, 0, this.internal, this.internal);
    }

    public getFloat32(): Float32Array {
        const imgData = this.ctx.getImageData(0, 0, this.internal, this.internal);
        const out = new Float32Array(this.internal * this.internal);
        for (let i = 0; i < out.length; i++) {
            out[i] = (imgData.data[i * 4] / 127.5) - 1.0;
        }
        return out;
    }

    public setFloat32(data: Float32Array) {
        const imgData = this.ctx.createImageData(this.internal, this.internal);
        for (let i = 0; i < data.length; i++) {
            const v = Math.max(0, Math.min(255, (data[i] + 1) * 127.5));
            imgData.data[i * 4]     = v;
            imgData.data[i * 4 + 1] = v;
            imgData.data[i * 4 + 2] = v;
            imgData.data[i * 4 + 3] = 255;
        }
        this.ctx.putImageData(imgData, 0, 0);
        this.hasContent = true;
    }
}
