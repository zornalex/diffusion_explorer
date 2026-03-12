import { CONFIG } from '../core/config';

export class DrawingCanvas {
    private canvas: HTMLCanvasElement;
    private ctx: CanvasRenderingContext2D;
    private isDrawing: boolean = false;

    constructor(canvasId: string) {
        const canvas = document.getElementById(canvasId) as HTMLCanvasElement;
        if (!canvas) throw new Error(`Canvas with id ${canvasId} not found`);

        this.canvas = canvas;
        this.ctx = canvas.getContext('2d', { willReadFrequently: true })!;

        // Set canvas size from config
        this.canvas.width = CONFIG.imageSize;
        this.canvas.height = CONFIG.imageSize;

        // Scale up visually with CSS, but keep internal resolution low
        this.canvas.style.width = '160px';
        this.canvas.style.height = '160px';
        this.canvas.style.imageRendering = 'pixelated';

        this.initEvents();
        this.clear();
    }

    private initEvents() {
        // Mouse Events
        this.canvas.addEventListener('mousedown', this.startDrawing.bind(this));
        this.canvas.addEventListener('mousemove', this.draw.bind(this));
        this.canvas.addEventListener('mouseup', this.stopDrawing.bind(this));
        this.canvas.addEventListener('mouseout', this.stopDrawing.bind(this));

        // Touch Events
        this.canvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            this.startDrawing(e.touches[0]);
        });
        this.canvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
            this.draw(e.touches[0]);
        });
        this.canvas.addEventListener('touchend', this.stopDrawing.bind(this));
    }

    private getPos(e: MouseEvent | Touch) {
        const rect = this.canvas.getBoundingClientRect();
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;
        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY
        };
    }

    private startDrawing(e: MouseEvent | Touch) {
        this.isDrawing = true;
        this.draw(e);
    }

    private draw(e: MouseEvent | Touch) {
        if (!this.isDrawing) return;

        const { x, y } = this.getPos(e);

        this.ctx.fillStyle = 'white';
        this.ctx.beginPath();
        this.ctx.arc(x, y, 3, 0, Math.PI * 2); // Brush size 3px
        this.ctx.fill();
    }

    private stopDrawing() {
        this.isDrawing = false;
    }

    public clear() {
        this.ctx.fillStyle = 'black';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    }

    public getImageData(): ImageData {
        return this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
    }

    public drawImage(img: HTMLImageElement) {
        // Draw image to fit canvas
        this.ctx.drawImage(img, 0, 0, this.canvas.width, this.canvas.height);

        // Convert to grayscale
        const imageData = this.getImageData();
        const data = imageData.data;

        for (let i = 0; i < data.length; i += 4) {
            const avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
            data[i] = avg;     // R
            data[i + 1] = avg; // G
            data[i + 2] = avg; // B
        }

        this.ctx.putImageData(imageData, 0, 0);
    }
}
