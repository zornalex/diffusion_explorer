/**
 * Preset smiley images as 32×32 Float32Arrays in [-1, 1] range.
 * Generated programmatically to keep zero network dependencies.
 */

const SIZE = 32;

function blank(): Float32Array {
    return new Float32Array(SIZE * SIZE).fill(-1.0);
}

function px(img: Float32Array, x: number, y: number, v: number = 1.0) {
    if (x < 0 || x >= SIZE || y < 0 || y >= SIZE) return;
    img[y * SIZE + x] = v;
}

function circle(img: Float32Array, cx: number, cy: number, r: number, fill = false) {
    for (let y = 0; y < SIZE; y++) {
        for (let x = 0; x < SIZE; x++) {
            const d = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);
            if (fill ? d <= r : Math.abs(d - r) < 1.2) {
                img[y * SIZE + x] = 0.9;
            }
        }
    }
}

function arc(img: Float32Array, cx: number, cy: number, r: number, fromDeg: number, toDeg: number) {
    for (let deg = fromDeg; deg <= toDeg; deg += 2) {
        const rad = (deg * Math.PI) / 180;
        const x = Math.round(cx + r * Math.cos(rad));
        const y = Math.round(cy + r * Math.sin(rad));
        px(img, x, y);
        px(img, x + 1, y);
        px(img, x, y + 1);
    }
}

function dot(img: Float32Array, cx: number, cy: number, r: number) {
    circle(img, cx, cy, r, true);
}

function hline(img: Float32Array, y: number, x0: number, x1: number) {
    for (let x = x0; x <= x1; x++) px(img, x, y);
}


/** Happy smiley :-) */
function makeSmileyHappy(): Float32Array {
    const img = blank();
    circle(img, 16, 16, 12);          // face outline
    dot(img, 12, 12, 1.5);             // left eye
    dot(img, 20, 12, 1.5);             // right eye
    arc(img, 16, 16, 6, 20, 160);     // smile curve (bottom arc)
    return img;
}

/** Neutral smiley :-| */
function makeSmileyNeutral(): Float32Array {
    const img = blank();
    circle(img, 16, 16, 12);
    dot(img, 12, 12, 1.5);
    dot(img, 20, 12, 1.5);
    hline(img, 22, 12, 20);            // flat mouth
    hline(img, 23, 12, 20);
    return img;
}

/** Sad smiley :-( */
function makeSmilelySad(): Float32Array {
    const img = blank();
    circle(img, 16, 16, 12);
    dot(img, 12, 12, 1.5);
    dot(img, 20, 12, 1.5);
    arc(img, 16, 28, 6, 200, 340);    // frown (upper arc, shifted down)
    return img;
}

/** Surprised smiley :-O */
function makeSmileySuprise(): Float32Array {
    const img = blank();
    circle(img, 16, 16, 12);
    dot(img, 12, 12, 1.5);
    dot(img, 20, 12, 1.5);
    circle(img, 16, 22, 3);            // round mouth
    return img;
}

/** Winking smiley ;-) */
function makeSmileyWink(): Float32Array {
    const img = blank();
    circle(img, 16, 16, 12);
    // left eye: closed (line)
    hline(img, 12, 10, 14);
    hline(img, 13, 10, 14);
    dot(img, 20, 12, 1.5);             // right eye open
    arc(img, 16, 16, 6, 20, 160);
    return img;
}

/** Cool smiley 8-) */
function makeSmiley8(): Float32Array {
    const img = blank();
    circle(img, 16, 16, 12);
    circle(img, 12, 12, 2.5);          // sunglasses left
    circle(img, 20, 12, 2.5);          // sunglasses right
    hline(img, 12, 14, 18);            // bridge
    arc(img, 16, 16, 6, 20, 160);
    return img;
}

export const SMILEY_PRESETS: Float32Array[] = [
    makeSmileyHappy(),
    makeSmileyHappy(),
    makeSmileyHappy(),
    makeSmileyNeutral(),
    makeSmileyNeutral(),
    makeSmilelySad(),
    makeSmilelySad(),
    makeSmileySuprise(),
];

export const SMILEY_LABELS: string[] = [
    '😊', '😊', '😊', '😐', '😐', '😢', '😢', '😮'
];

/** Three class presets: happy / neutral / sad (for conditional generation demo) */
export const CLASS_PRESETS: Float32Array[][] = [
    [makeSmileyHappy(), makeSmileyHappy(), makeSmileyWink()],     // class 0: happy
    [makeSmileyNeutral(), makeSmiley8()],                          // class 1: neutral
    [makeSmilelySad(), makeSmilelySad()],                          // class 2: sad
];

export const CLASS_LABELS = ['😊 Happy', '😐 Neutral', '😢 Sad'];
