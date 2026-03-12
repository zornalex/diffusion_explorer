export class LossChart {
    private canvas: HTMLCanvasElement;
    private ctx: CanvasRenderingContext2D;
    private history: number[] = [];
    private readonly movingAvgWindow = 10;
    private readonly maxLoss = 0.5;

    constructor(canvasId: string) {
        this.canvas = document.getElementById(canvasId) as HTMLCanvasElement;
        this.ctx = this.canvas.getContext('2d')!;
        this.clear();
    }

    public addPoint(loss: number): void {
        this.history.push(loss);
        this.render();
    }

    public reset(): void {
        this.history = [];
        this.clear();
    }

    private clear(): void {
        const { ctx, canvas } = this;
        ctx.fillStyle = '#0d0d0d';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    }

    private render(): void {
        const { ctx, canvas, history, maxLoss, movingAvgWindow } = this;
        const w = canvas.width;
        const h = canvas.height;

        this.clear();

        if (history.length < 2) return;

        // Grid lines
        ctx.strokeStyle = '#222';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 4; i++) {
            const y = (i / 4) * h;
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(w, y);
            ctx.stroke();
        }

        // Y-axis labels
        ctx.fillStyle = '#555';
        ctx.font = '10px monospace';
        ctx.fillText('0.5', 4, 11);
        ctx.fillText('0.0', 4, h - 3);

        // Raw loss (subtle)
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 1;
        ctx.beginPath();
        history.forEach((loss, i) => {
            const x = (i / Math.max(history.length - 1, 1)) * w;
            const y = h - Math.min(loss / maxLoss, 1) * h;
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.stroke();

        // Moving average (bright)
        if (history.length >= movingAvgWindow) {
            ctx.strokeStyle = '#818cf8';
            ctx.lineWidth = 2;
            ctx.beginPath();
            for (let i = movingAvgWindow - 1; i < history.length; i++) {
                const window = history.slice(i - movingAvgWindow + 1, i + 1);
                const avg = window.reduce((a, b) => a + b, 0) / movingAvgWindow;
                const x = (i / Math.max(history.length - 1, 1)) * w;
                const y = h - Math.min(avg / maxLoss, 1) * h;
                i === movingAvgWindow - 1 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
            }
            ctx.stroke();
        }
    }
}
