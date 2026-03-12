/**
 * Robotics & Flow Matching animated demos.
 * Purely visual — no ML model, just animated canvas.
 */

// ─── Robotics path demo ───────────────────────────────────────────────────

export class RoboticsDemo {
    private canvas: HTMLCanvasElement;
    private ctx: CanvasRenderingContext2D;
    private animFrame: number | null = null;
    private step = 0;
    private totalSteps = 60;

    // Fixed obstacles and waypoints for the demo
    private obstacles = [
        { x: 140, y: 120, r: 25 },
        { x: 260, y: 200, r: 30 },
        { x: 180, y: 280, r: 20 },
    ];
    private start = { x: 50,  y: 50  };
    private goal  = { x: 350, y: 320 };

    constructor(canvasId: string) {
        this.canvas = document.getElementById(canvasId) as HTMLCanvasElement;
        this.ctx = this.canvas.getContext('2d')!;
        this.drawStatic(1.0); // show final path immediately
    }

    /** Animate: noisy scribble path → smooth optimized path */
    public play() {
        if (this.animFrame !== null) return;
        this.step = 0;
        const tick = () => {
            this.step++;
            const progress = Math.min(this.step / this.totalSteps, 1);
            this.drawStatic(progress);
            if (this.step < this.totalSteps + 10) {
                this.animFrame = requestAnimationFrame(tick);
            } else {
                this.animFrame = null;
            }
        };
        tick();
    }

    public stop() {
        if (this.animFrame !== null) { cancelAnimationFrame(this.animFrame); this.animFrame = null; }
    }

    private drawStatic(progress: number) {
        const { ctx, canvas, obstacles, start, goal } = this;
        const W = canvas.width, H = canvas.height;
        ctx.clearRect(0, 0, W, H);

        // Background grid
        ctx.strokeStyle = '#1e1e2e';
        ctx.lineWidth = 1;
        for (let x = 0; x < W; x += 20) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
        for (let y = 0; y < H; y += 20) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }

        // Obstacles
        obstacles.forEach(o => {
            ctx.beginPath();
            ctx.arc(o.x, o.y, o.r, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(251, 146, 60, 0.15)';
            ctx.fill();
            ctx.strokeStyle = '#fb923c';
            ctx.lineWidth = 1.5;
            ctx.stroke();
        });

        // Noisy path (fades out as progress → 1)
        const noiseAlpha = Math.max(0, 1 - progress * 2);
        if (noiseAlpha > 0) {
            ctx.save();
            ctx.globalAlpha = noiseAlpha * 0.6;
            ctx.strokeStyle = '#64748b';
            ctx.lineWidth = 1.5;
            ctx.setLineDash([3, 3]);
            ctx.beginPath();
            ctx.moveTo(start.x, start.y);
            // Noisy waypoints
            const noisy = this.getNoisyWaypoints(12);
            noisy.forEach(p => ctx.lineTo(p.x, p.y));
            ctx.lineTo(goal.x, goal.y);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.restore();
        }

        // Smooth path (appears as progress → 1)
        const smoothAlpha = Math.min(progress * 1.5, 1);
        if (smoothAlpha > 0) {
            const smoothPts = this.getSmoothPath();
            // Glow
            ctx.save();
            ctx.globalAlpha = smoothAlpha * 0.25;
            ctx.strokeStyle = '#818cf8';
            ctx.lineWidth = 8;
            ctx.lineJoin = 'round';
            this.drawPath(smoothPts, progress);
            ctx.stroke();
            ctx.restore();
            // Main line
            ctx.save();
            ctx.globalAlpha = smoothAlpha;
            ctx.strokeStyle = '#818cf8';
            ctx.lineWidth = 2.5;
            ctx.lineJoin = 'round';
            this.drawPath(smoothPts, progress);
            ctx.stroke();
            ctx.restore();
        }

        // Start & Goal
        this.drawPoint(start.x, start.y, '#4ade80', 'Start');
        this.drawPoint(goal.x,  goal.y,  '#c084fc', 'Goal');
    }

    private drawPath(pts: Array<{x:number;y:number}>, progress: number) {
        const n = Math.min(Math.floor(pts.length * progress), pts.length - 1);
        this.ctx.beginPath();
        this.ctx.moveTo(pts[0].x, pts[0].y);
        for (let i = 1; i <= n; i++) this.ctx.lineTo(pts[i].x, pts[i].y);
    }

    private drawPoint(x: number, y: number, color: string, label: string) {
        const ctx = this.ctx;
        ctx.beginPath();
        ctx.arc(x, y, 7, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.font = '11px Inter, sans-serif';
        ctx.fillStyle = color;
        ctx.fillText(label, x + 10, y + 4);
    }

    private getNoisyWaypoints(n: number) {
        const pts = [];
        const dx = (this.goal.x - this.start.x) / (n + 1);
        const dy = (this.goal.y - this.start.y) / (n + 1);
        // Seed random for consistent look
        let seed = 42;
        const rand = () => { seed = (seed * 16807) % 2147483647; return (seed / 2147483647) - 0.5; };
        for (let i = 1; i <= n; i++) {
            pts.push({
                x: this.start.x + dx * i + rand() * 60,
                y: this.start.y + dy * i + rand() * 60,
            });
        }
        return pts;
    }

    private getSmoothPath() {
        // Bezier-approximated smooth path avoiding obstacles
        return [
            this.start,
            { x: 90,  y: 80  },
            { x: 110, y: 160 },
            { x: 210, y: 150 },
            { x: 300, y: 160 },
            { x: 310, y: 240 },
            { x: 290, y: 290 },
            this.goal,
        ];
    }
}

// ─── Flow Matching comparison demo ────────────────────────────────────────

export class FlowMatchingDemo {
    private canvas: HTMLCanvasElement;
    private ctx: CanvasRenderingContext2D;
    private animFrame: number | null = null;
    private step = 0;
    private readonly totalSteps = 80;

    constructor(canvasId: string) {
        this.canvas = document.getElementById(canvasId) as HTMLCanvasElement;
        this.ctx = this.canvas.getContext('2d')!;
        this.drawFrame(0);
    }

    public play() {
        if (this.animFrame !== null) return;
        this.step = 0;
        const tick = () => {
            this.step++;
            this.drawFrame(this.step / this.totalSteps);
            if (this.step < this.totalSteps) {
                this.animFrame = requestAnimationFrame(tick);
            } else {
                this.animFrame = null;
                // Loop
                setTimeout(() => { if (this.animFrame === null) this.play(); }, 2000);
            }
        };
        tick();
    }

    public stop() {
        if (this.animFrame !== null) { cancelAnimationFrame(this.animFrame); this.animFrame = null; }
    }

    private drawFrame(progress: number) {
        const { ctx, canvas } = this;
        const W = canvas.width, H = canvas.height;
        ctx.clearRect(0, 0, W, H);

        const half = W / 2;

        // ── Left: DDPM (zigzag many steps) ──────────────────────────────
        ctx.save();
        ctx.beginPath();
        ctx.rect(0, 0, half - 1, H);
        ctx.clip();

        this.drawPanelBg(0, 0, half, H, 'DDPM');

        const noiseL = { x: 40, y: 60 };
        const imageL = { x: half - 50, y: H - 60 };

        // Draw zigzag path
        const zigzagPts = this.getZigzagPath(noiseL, imageL, 12);
        const nZ = Math.floor(zigzagPts.length * progress);
        if (nZ > 1) {
            ctx.strokeStyle = '#818cf8';
            ctx.lineWidth = 1.5;
            ctx.setLineDash([]);
            ctx.beginPath();
            ctx.moveTo(zigzagPts[0].x, zigzagPts[0].y);
            for (let i = 1; i < nZ; i++) ctx.lineTo(zigzagPts[i].x, zigzagPts[i].y);
            ctx.stroke();

            // Current dot
            const cur = zigzagPts[nZ - 1];
            ctx.beginPath();
            ctx.arc(cur.x, cur.y, 4, 0, Math.PI * 2);
            ctx.fillStyle = '#818cf8';
            ctx.fill();
        }

        this.drawEndpoints(noiseL, imageL);
        ctx.restore();

        // Divider
        ctx.strokeStyle = '#252525';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(half, 0);
        ctx.lineTo(half, H);
        ctx.stroke();

        // ── Right: Flow Matching (straight) ─────────────────────────────
        ctx.save();
        ctx.beginPath();
        ctx.rect(half + 1, 0, half, H);
        ctx.clip();

        this.drawPanelBg(half, 0, half, H, 'Flow Matching');

        const noiseR = { x: half + 40, y: 60 };
        const imageR = { x: W - 50, y: H - 60 };

        // Draw straight path
        ctx.strokeStyle = '#4ade80';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(noiseR.x, noiseR.y);
        const ex = noiseR.x + (imageR.x - noiseR.x) * progress;
        const ey = noiseR.y + (imageR.y - noiseR.y) * progress;
        ctx.lineTo(ex, ey);
        ctx.stroke();

        // Current dot
        if (progress > 0) {
            ctx.beginPath();
            ctx.arc(ex, ey, 5, 0, Math.PI * 2);
            ctx.fillStyle = '#4ade80';
            ctx.fill();
        }

        this.drawEndpoints(noiseR, imageR);
        ctx.restore();
    }

    private drawPanelBg(x: number, y: number, w: number, h: number, label: string) {
        const ctx = this.ctx;
        // Grid
        ctx.strokeStyle = '#1a1a2e';
        ctx.lineWidth = 1;
        for (let gx = x; gx < x + w; gx += 25) {
            ctx.beginPath(); ctx.moveTo(gx, y); ctx.lineTo(gx, y + h); ctx.stroke();
        }
        for (let gy = y; gy < y + h; gy += 25) {
            ctx.beginPath(); ctx.moveTo(x, gy); ctx.lineTo(x + w, gy); ctx.stroke();
        }
        // Label
        ctx.font = 'bold 11px Inter, sans-serif';
        ctx.fillStyle = '#334155';
        ctx.fillText(label, x + 8, y + 18);
    }

    private drawEndpoints(from: {x:number;y:number}, to: {x:number;y:number}) {
        const ctx = this.ctx;
        // Noise cloud (from)
        ctx.beginPath(); ctx.arc(from.x, from.y, 8, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(100,116,139,0.3)'; ctx.fill();
        ctx.strokeStyle = '#64748b'; ctx.lineWidth = 1.5; ctx.stroke();
        ctx.font = '10px Inter, sans-serif'; ctx.fillStyle = '#64748b';
        ctx.fillText('noise', from.x - 14, from.y - 12);

        // Image (to)
        ctx.beginPath(); ctx.arc(to.x, to.y, 8, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(192,132,252,0.3)'; ctx.fill();
        ctx.strokeStyle = '#c084fc'; ctx.lineWidth = 1.5; ctx.stroke();
        ctx.font = '10px Inter, sans-serif'; ctx.fillStyle = '#c084fc';
        ctx.fillText('image', to.x - 14, to.y + 18);
    }

    private getZigzagPath(from: {x:number;y:number}, to: {x:number;y:number}, n: number) {
        const pts: Array<{x:number;y:number}> = [from];
        const dx = (to.x - from.x) / n;
        const dy = (to.y - from.y) / n;
        let seed = 7;
        const rand = () => { seed = (seed * 16807) % 2147483647; return (seed / 2147483647 - 0.5) * 2; };
        for (let i = 1; i < n; i++) {
            pts.push({
                x: from.x + dx * i + rand() * 35,
                y: from.y + dy * i + rand() * 35,
            });
        }
        pts.push(to);
        return pts;
    }
}
