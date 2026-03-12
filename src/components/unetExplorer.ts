/**
 * Interactive U-Net Architecture Explorer.
 * Canvas-based diagram with clickable blocks that show detail popups.
 * Shows the exact architecture from trainer.ts: 32/64/128 filters, skip connections.
 */

interface UNetBlock {
    id: string;
    x: number;
    y: number;
    w: number;
    h: number;
    label: string;     // shown inside block
    sublabel: string;  // shape shown below label
    color: string;     // block fill color
    detail: BlockDetail;
}

interface BlockDetail {
    title: string;
    operation: string;
    shape_in: string;
    shape_out: string;
    activation: string;
    note?: string;
}

interface UNetArrow {
    fromX: number; fromY: number;
    toX: number;   toY: number;
    dashed?: boolean;
    label?: string;
    color?: string;
}

// ─── Color palette (dark theme) ────────────────────────────────────────────
const C_ENCODER    = '#312e81'; // deep indigo
const C_ENCODER_B  = '#818cf8'; // indigo-400 (border)
const C_BOTTLENECK = '#4c1d95'; // deep purple
const C_BOTTLENECK_B = '#c084fc'; // purple-400
const C_DECODER    = '#052e16'; // deep green
const C_DECODER_B  = '#4ade80'; // green-400
const C_INPUT      = '#1e3a5f'; // blue
const C_INPUT_B    = '#60a5fa'; // blue-400
const C_OUTPUT     = '#7c2d12'; // orange
const C_OUTPUT_B   = '#fb923c'; // orange-400
const C_TIME       = '#1c2a1c'; // dark green
const C_TIME_B     = '#86efac'; // green-300
// (skip color defined inline in arrows)

// Canvas dimensions
const CW = 520;
const CH = 490;

// Block layout constants
const BW = 160;    // block width
const BH = 40;     // block height
const CX = CW / 2; // center x

export class UNetExplorer {
    private canvas: HTMLCanvasElement;
    private ctx: CanvasRenderingContext2D;
    private popup: HTMLElement;
    private blocks: UNetBlock[];
    private arrows: UNetArrow[];
    private activeId: string | null = null;

    constructor(canvasId: string, popupId: string) {
        this.canvas = document.getElementById(canvasId) as HTMLCanvasElement;
        this.canvas.width = CW;
        this.canvas.height = CH;
        this.canvas.style.width = `${CW}px`;
        this.canvas.style.height = `${CH}px`;
        this.ctx = this.canvas.getContext('2d')!;

        this.popup = document.getElementById(popupId) as HTMLElement;
        this.blocks = this.defineBlocks();
        this.arrows = this.defineArrows();

        this.canvas.addEventListener('click', (e) => this.onClick(e));
        this.canvas.style.cursor = 'pointer';

        this.render();
    }

    // ─── Block definitions ──────────────────────────────────────────────────

    private defineBlocks(): UNetBlock[] {
        // Y positions for each row
        const y0 = 15;   // input row
        const y1 = 80;   // encoder L1
        const y2 = 160;  // encoder L2
        const y3 = 240;  // bottleneck
        const y4 = 320;  // decoder L2
        const y5 = 400;  // decoder L1
        const y6 = 460;  // output

        return [
            // ── Inputs ──────────────────────────────────────────────────
            {
                id: 'input-img',
                x: CX - BW / 2 - 50, y: y0, w: BW - 30, h: BH,
                label: 'xₜ  (noisy image)',
                sublabel: '32×32×1',
                color: C_INPUT,
                detail: {
                    title: 'Input: Noisy Image xₜ',
                    operation: 'Input tensor',
                    shape_in: '—',
                    shape_out: '[1, 32, 32, 1]',
                    activation: '—',
                    note: 'Pixel values normalized to [-1, 1]. Combined with time embedding via concatenation.',
                },
            },
            {
                id: 'input-time',
                x: CX + 60, y: y0, w: BW - 50, h: BH,
                label: 't / T',
                sublabel: '[1]',
                color: C_TIME,
                detail: {
                    title: 'Input: Timestep t',
                    operation: 'Dense(1024, relu) → Reshape(32×32×1)',
                    shape_in: '[1, 1]',
                    shape_out: '[1, 32, 32, 1]',
                    activation: 'ReLU',
                    note: 'Scalar timestep t/T ∈ [0,1] projected to a full spatial map, then concatenated with xₜ.',
                },
            },

            // ── Encoder ─────────────────────────────────────────────────
            {
                id: 'enc1',
                x: CX - BW / 2, y: y1, w: BW, h: BH,
                label: 'Conv32 × 2',
                sublabel: '32×32×32',
                color: C_ENCODER,
                detail: {
                    title: 'Encoder Block 1',
                    operation: 'Conv2D(32, 3×3, same) × 2 + MaxPool(2×2)',
                    shape_in: '[1, 32, 32, 2]',
                    shape_out: '[1, 16, 16, 32]  (after pool)',
                    activation: 'ReLU (heNormal init)',
                    note: 'Skip connection saved before pooling: 32×32×32 tensor forwarded to Decoder Block 1.',
                },
            },
            {
                id: 'enc2',
                x: CX - BW / 2, y: y2, w: BW, h: BH,
                label: 'Conv64 × 2',
                sublabel: '16×16×64',
                color: C_ENCODER,
                detail: {
                    title: 'Encoder Block 2',
                    operation: 'Conv2D(64, 3×3, same) × 2 + MaxPool(2×2)',
                    shape_in: '[1, 16, 16, 32]',
                    shape_out: '[1, 8, 8, 64]  (after pool)',
                    activation: 'ReLU (heNormal init)',
                    note: 'Skip connection saved before pooling: 16×16×64 tensor forwarded to Decoder Block 2.',
                },
            },

            // ── Bottleneck ───────────────────────────────────────────────
            {
                id: 'bottleneck',
                x: CX - BW / 2, y: y3, w: BW, h: BH,
                label: 'Conv128 × 2',
                sublabel: '8×8×128',
                color: C_BOTTLENECK,
                detail: {
                    title: 'Bottleneck',
                    operation: 'Conv2D(128, 3×3, same) × 2',
                    shape_in: '[1, 8, 8, 64]',
                    shape_out: '[1, 8, 8, 128]',
                    activation: 'ReLU (heNormal init)',
                    note: 'Smallest spatial resolution. Captures global context. No skip from here — outputs feed directly into Decoder.',
                },
            },

            // ── Decoder ─────────────────────────────────────────────────
            {
                id: 'dec2',
                x: CX - BW / 2, y: y4, w: BW, h: BH,
                label: 'UpSample + Conv64 × 2',
                sublabel: '16×16×64',
                color: C_DECODER,
                detail: {
                    title: 'Decoder Block 2',
                    operation: 'UpSampling2D(2×2) → Concat(skip2) → Conv2D(64, 3×3) × 2',
                    shape_in: '[1, 8, 8, 128]',
                    shape_out: '[1, 16, 16, 64]',
                    activation: 'ReLU (heNormal init)',
                    note: 'Concatenates with Encoder 2 skip (16×16×64). Merged shape: 16×16×192 → 16×16×64.',
                },
            },
            {
                id: 'dec1',
                x: CX - BW / 2, y: y5, w: BW, h: BH,
                label: 'UpSample + Conv32',
                sublabel: '32×32×32',
                color: C_DECODER,
                detail: {
                    title: 'Decoder Block 1',
                    operation: 'UpSampling2D(2×2) → Concat(skip1) → Conv2D(32, 3×3)',
                    shape_in: '[1, 16, 16, 64]',
                    shape_out: '[1, 32, 32, 32]',
                    activation: 'ReLU (heNormal init)',
                    note: 'Concatenates with Encoder 1 skip (32×32×32). Merged shape: 32×32×96 → 32×32×32.',
                },
            },

            // ── Output ───────────────────────────────────────────────────
            {
                id: 'output',
                x: CX - BW / 2, y: y6, w: BW, h: BH - 10,
                label: 'Conv1×1  →  ε̂',
                sublabel: '32×32×1',
                color: C_OUTPUT,
                detail: {
                    title: 'Output: Predicted Noise ε̂',
                    operation: 'Conv2D(1, 1×1, same, zeros-init)',
                    shape_in: '[1, 32, 32, 32]',
                    shape_out: '[1, 32, 32, 1]',
                    activation: 'Linear (no activation)',
                    note: 'Predicts the noise ε that was added to x₀. Training loss: MSE(ε̂, ε). Kernel initialized to zeros for stable early training.',
                },
            },
        ];
    }

    // ─── Arrow definitions ─────────────────────────────────────────────────

    private defineArrows(): UNetArrow[] {
        const block = (id: string) => this.blocks.find(b => b.id === id)!;

        const bottomCenter = (b: UNetBlock) => ({ x: b.x + b.w / 2, y: b.y + b.h });
        const topCenter    = (b: UNetBlock) => ({ x: b.x + b.w / 2, y: b.y });
        const rightEdge    = (b: UNetBlock) => ({ x: b.x + b.w, y: b.y + b.h / 2 });
        const _leftEdge    = (b: UNetBlock) => ({ x: b.x, y: b.y + b.h / 2 }); void _leftEdge;

        const arrows: UNetArrow[] = [];

        // Input → Enc1 (both inputs merge into enc1)
        const imgB    = block('input-img');
        const timeB   = block('input-time');
        const enc1B   = block('enc1');
        const enc2B   = block('enc2');
        const botB    = block('bottleneck');
        const dec2B   = block('dec2');
        const dec1B   = block('dec1');
        const outB    = block('output');

        const catX    = CX;              // concat junction X
        const catY    = enc1B.y - 12;   // concat junction Y (above enc1)

        // img → concat
        arrows.push({
            fromX: bottomCenter(imgB).x, fromY: bottomCenter(imgB).y,
            toX:   catX - 6,            toY:   catY,
        });
        // time → concat
        arrows.push({
            fromX: bottomCenter(timeB).x, fromY: bottomCenter(timeB).y,
            toX:   catX + 6,             toY:   catY,
        });
        // concat → enc1
        arrows.push({
            fromX: catX, fromY: catY,
            toX:   topCenter(enc1B).x, toY: topCenter(enc1B).y,
        });

        // enc1 → enc2
        arrows.push({ fromX: bottomCenter(enc1B).x, fromY: bottomCenter(enc1B).y, toX: topCenter(enc2B).x, toY: topCenter(enc2B).y });

        // enc2 → bottleneck
        arrows.push({ fromX: bottomCenter(enc2B).x, fromY: bottomCenter(enc2B).y, toX: topCenter(botB).x, toY: topCenter(botB).y });

        // bottleneck → dec2
        arrows.push({ fromX: bottomCenter(botB).x, fromY: bottomCenter(botB).y, toX: topCenter(dec2B).x, toY: topCenter(dec2B).y });

        // dec2 → dec1
        arrows.push({ fromX: bottomCenter(dec2B).x, fromY: bottomCenter(dec2B).y, toX: topCenter(dec1B).x, toY: topCenter(dec1B).y });

        // dec1 → output
        arrows.push({ fromX: bottomCenter(dec1B).x, fromY: bottomCenter(dec1B).y, toX: topCenter(outB).x, toY: topCenter(outB).y });

        // ── Skip connections (dashed, right side) ──────────────────────
        const skipX1 = enc1B.x + enc1B.w + 20;    // skip1 goes right of enc1
        const skipX2 = enc2B.x + enc2B.w + 50;    // skip2 goes farther right

        // Skip 1: enc1 right → loop right → dec1 right
        arrows.push({
            fromX: rightEdge(enc1B).x,  fromY: rightEdge(enc1B).y,
            toX:   skipX1,              toY:   rightEdge(enc1B).y,
            dashed: true, color: C_ENCODER_B,
        });
        arrows.push({
            fromX: skipX1, fromY: rightEdge(enc1B).y,
            toX:   skipX1, toY:   rightEdge(dec1B).y,
            dashed: true, color: C_ENCODER_B,
        });
        arrows.push({
            fromX: skipX1,              fromY: rightEdge(dec1B).y,
            toX:   rightEdge(dec1B).x,  toY:   rightEdge(dec1B).y,
            dashed: true, color: C_ENCODER_B, label: 'skip1',
        });

        // Skip 2: enc2 right → loop right → dec2 right
        arrows.push({
            fromX: rightEdge(enc2B).x,  fromY: rightEdge(enc2B).y,
            toX:   skipX2,              toY:   rightEdge(enc2B).y,
            dashed: true, color: C_DECODER_B,
        });
        arrows.push({
            fromX: skipX2, fromY: rightEdge(enc2B).y,
            toX:   skipX2, toY:   rightEdge(dec2B).y,
            dashed: true, color: C_DECODER_B,
        });
        arrows.push({
            fromX: skipX2,              fromY: rightEdge(dec2B).y,
            toX:   rightEdge(dec2B).x,  toY:   rightEdge(dec2B).y,
            dashed: true, color: C_DECODER_B, label: 'skip2',
        });

        return arrows;
    }

    // ─── Rendering ─────────────────────────────────────────────────────────

    private render() {
        const { ctx } = this;
        ctx.clearRect(0, 0, CW, CH);

        // Background
        ctx.fillStyle = '#0d0d0d';
        ctx.fillRect(0, 0, CW, CH);

        this.drawArrows();
        this.drawBlocks();
        this.drawConcatDot();
    }

    private drawArrows() {
        const { ctx } = this;

        for (const a of this.arrows) {
            const color = a.color ?? '#475569';
            ctx.strokeStyle = color;
            ctx.lineWidth = 1.5;
            if (a.dashed) {
                ctx.setLineDash([4, 3]);
            } else {
                ctx.setLineDash([]);
            }

            ctx.beginPath();
            ctx.moveTo(a.fromX, a.fromY);
            ctx.lineTo(a.toX, a.toY);
            ctx.stroke();
            ctx.setLineDash([]);

            // Arrowhead on final segment of skip connections (pointing left, into decoder)
            if (a.dashed && a.label) {
                this.drawArrowhead(a.toX, a.toY, 'left', color);
                // Label
                ctx.fillStyle = color;
                ctx.font = '9px Inter, sans-serif';
                ctx.fillText(a.label, a.fromX + 3, a.fromY - 4);
            }
            // Arrowhead on main flow arrows (pointing down)
            if (!a.dashed) {
                this.drawArrowhead(a.toX, a.toY, 'down', color);
            }
        }
    }

    private drawArrowhead(x: number, y: number, dir: 'down' | 'left', color: string) {
        const { ctx } = this;
        const s = 5;
        ctx.fillStyle = color;
        ctx.beginPath();
        if (dir === 'down') {
            ctx.moveTo(x, y); ctx.lineTo(x - s, y - s * 1.5); ctx.lineTo(x + s, y - s * 1.5);
        } else {
            ctx.moveTo(x, y); ctx.lineTo(x + s * 1.5, y - s); ctx.lineTo(x + s * 1.5, y + s);
        }
        ctx.closePath();
        ctx.fill();
    }

    private drawConcatDot() {
        const enc1B = this.blocks.find(b => b.id === 'enc1')!;
        const catY  = enc1B.y - 12;
        const ctx = this.ctx;
        ctx.beginPath();
        ctx.arc(CX, catY, 4, 0, Math.PI * 2);
        ctx.fillStyle = '#94a3b8';
        ctx.fill();
        ctx.fillStyle = '#475569';
        ctx.font = '9px Inter, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('concat', CX + 6, catY + 3);
        ctx.textAlign = 'left';
    }

    private drawBlocks() {
        for (const block of this.blocks) {
            this.drawBlock(block);
        }
    }

    private drawBlock(b: UNetBlock) {
        const { ctx } = this;
        const isActive = b.id === this.activeId;

        // Determine border color from block color family
        let borderColor: string;
        if      (b.id.startsWith('enc'))   borderColor = C_ENCODER_B;
        else if (b.id === 'bottleneck')    borderColor = C_BOTTLENECK_B;
        else if (b.id.startsWith('dec'))   borderColor = C_DECODER_B;
        else if (b.id.startsWith('input')) borderColor = b.id === 'input-time' ? C_TIME_B : C_INPUT_B;
        else                               borderColor = C_OUTPUT_B;

        // Fill
        ctx.fillStyle = b.color;
        ctx.strokeStyle = isActive ? '#fff' : borderColor;
        ctx.lineWidth = isActive ? 2 : 1;
        ctx.beginPath();
        ctx.roundRect(b.x, b.y, b.w, b.h, 6);
        ctx.fill();
        ctx.stroke();

        // Label
        ctx.fillStyle = isActive ? '#fff' : borderColor;
        ctx.font = `bold 10px 'JetBrains Mono', monospace`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(b.label, b.x + b.w / 2, b.y + b.h / 2 - 6);

        // Sublabel (shape)
        ctx.fillStyle = '#64748b';
        ctx.font = '9px monospace';
        ctx.fillText(b.sublabel, b.x + b.w / 2, b.y + b.h / 2 + 7);

        ctx.textAlign = 'left';
        ctx.textBaseline = 'alphabetic';
    }

    // ─── Click handling ────────────────────────────────────────────────────

    private onClick(e: MouseEvent) {
        const rect = this.canvas.getBoundingClientRect();
        const scaleX = CW / rect.width;
        const scaleY = CH / rect.height;
        const mx = (e.clientX - rect.left) * scaleX;
        const my = (e.clientY - rect.top) * scaleY;

        const hit = this.blocks.find(
            b => mx >= b.x && mx <= b.x + b.w && my >= b.y && my <= b.y + b.h
        );

        if (hit) {
            this.activeId = hit.id === this.activeId ? null : hit.id;
            this.render();
            if (this.activeId) {
                this.showPopup(hit, e);
            } else {
                this.hidePopup();
            }
        } else {
            this.activeId = null;
            this.render();
            this.hidePopup();
        }
    }

    private showPopup(block: UNetBlock, e: MouseEvent) {
        const d = block.detail;
        this.popup.innerHTML = `
            <div class="unet-popup-close" id="unet-popup-close">×</div>
            <div class="unet-popup-title">${d.title}</div>
            <div class="unet-popup-row"><span class="unet-popup-key">Operation:</span> ${d.operation}</div>
            <div class="unet-popup-row"><span class="unet-popup-key">Input shape:</span> ${d.shape_in}</div>
            <div class="unet-popup-row"><span class="unet-popup-key">Output shape:</span> ${d.shape_out}</div>
            <div class="unet-popup-row"><span class="unet-popup-key">Activation:</span> ${d.activation}</div>
            ${d.note ? `<div class="unet-popup-note">${d.note}</div>` : ''}
        `;

        // Position popup near the click, inside the canvas wrapper
        const wrap = this.canvas.parentElement!;
        const wrapRect = wrap.getBoundingClientRect();
        let left = e.clientX - wrapRect.left + 8;
        let top  = e.clientY - wrapRect.top  + 8;

        // Keep inside wrap bounds (approximate)
        if (left + 260 > wrap.offsetWidth) left = e.clientX - wrapRect.left - 268;
        if (top  + 160 > wrap.offsetHeight) top  = e.clientY - wrapRect.top  - 168;

        this.popup.style.left = `${left}px`;
        this.popup.style.top  = `${top}px`;
        this.popup.classList.add('visible');

        // Close button
        document.getElementById('unet-popup-close')?.addEventListener('click', () => {
            this.activeId = null;
            this.render();
            this.hidePopup();
        });
    }

    private hidePopup() {
        this.popup.classList.remove('visible');
    }
}
