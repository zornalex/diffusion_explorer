import './style.css';
import { APP_VERSION, CONFIG }             from './core/config';
import { DiffusionModel }                  from './core/diffusion';
import { ModelTrainer }                    from './core/trainer';
import { LossChart }                       from './components/lossChart';
import { renderImageToCanvas, setupCanvas } from './components/renderUtils';
import { DatasetPanel }                    from './components/datasetPanel';
import { DenoisingStrip }                  from './components/denoisingStrip';
import { SMILEY_PRESETS }                  from './data/presets';
import { RoboticsDemo, FlowMatchingDemo }  from './components/roboticsDemo';
import { ForwardProcessDemo }              from './components/forwardDemo';
import { UNetExplorer }                    from './components/unetExplorer';

// ─── Core instances ─────────────────────────────────────────────────────────
const diffusion    = new DiffusionModel();
const trainer      = new ModelTrainer();
const lossChart    = new LossChart('loss-chart');
const datasetPanel = new DatasetPanel('dataset-slots', 5);
const strip        = new DenoisingStrip('denoising-strip', 'zoom-panel');

trainer.init().then(() => console.log('Trainer ready'));

// ─── Canvas contexts ─────────────────────────────────────────────────────────
const heroCleanCtx = setupCanvas('hero-canvas-clean',       CONFIG.imageSize, '80px');
const heroNoisyCtx = setupCanvas('hero-canvas-noisy',       CONFIG.imageSize, '80px');
const heroPureCtx  = setupCanvas('hero-canvas-pure',        CONFIG.imageSize, '80px');
const genIntermCtx = setupCanvas('gen-intermediate-canvas', CONFIG.imageSize, '160px');
const genCtx       = setupCanvas('gen-canvas',              CONFIG.imageSize, '160px');

function initDebugCanvas(id: string): CanvasRenderingContext2D {
    const c = document.getElementById(id) as HTMLCanvasElement;
    c.width = c.height = CONFIG.imageSize;
    return c.getContext('2d')!;
}

// ─── UI refs ─────────────────────────────────────────────────────────────────
const btnLoadPresets   = document.getElementById('btn-load-presets')   as HTMLButtonElement;
const btnClearDataset  = document.getElementById('btn-clear-dataset')  as HTMLButtonElement;
const btnTrain         = document.getElementById('btn-train')          as HTMLButtonElement;
const btnSaveModel     = document.getElementById('btn-save-model')     as HTMLButtonElement;
const btnLoadModel     = document.getElementById('btn-load-model')     as HTMLButtonElement;
const fileLoadModel    = document.getElementById('file-load-model')    as HTMLInputElement;
const btnGenerate      = document.getElementById('btn-generate')       as HTMLButtonElement;
const epochValue       = document.getElementById('epoch-value')        as HTMLSpanElement;
const maxEpochsEl      = document.getElementById('max-epochs')         as HTMLSpanElement;
const lossValueEl      = document.getElementById('loss-value')         as HTMLSpanElement;
const debugT           = document.getElementById('debug-t')            as HTMLSpanElement;
const genProgressSlider = document.getElementById('gen-progress-slider') as HTMLInputElement;
const genProgressValue  = document.getElementById('gen-progress-value') as HTMLSpanElement;
const genProgressMax    = document.getElementById('gen-progress-max')  as HTMLSpanElement;
const progressFill     = document.getElementById('training-progress-fill') as HTMLElement;
const progressLabel    = document.getElementById('training-progress-label') as HTMLElement;
const btnRoboticsPlay  = document.getElementById('btn-robotics-play')  as HTMLButtonElement;
const btnFlowPlay      = document.getElementById('btn-flow-play')      as HTMLButtonElement;

// ─── Init ─────────────────────────────────────────────────────────────────────
// ─── Version indicator ───────────────────────────────────────────────────────
(document.getElementById('app-version') as HTMLSpanElement).textContent = `v${APP_VERSION}`;

const T = diffusion.getTimesteps();
maxEpochsEl.textContent       = CONFIG.trainingSteps.toString();
genProgressSlider.max         = T.toString();
genProgressMax.textContent    = T.toString();

// ─── Hero preview ─────────────────────────────────────────────────────────────
function renderHeroPreview() {
    const S = CONFIG.imageSize;
    const clean = new Float32Array(S * S);
    const cx = S / 2, cy = S / 2, r = S * 0.3;
    // Draw circle
    for (let y = 0; y < S; y++)
        for (let x = 0; x < S; x++) {
            const d = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);
            clean[y * S + x] = Math.abs(d - r) < 1.5 ? 0.9 : -0.9;
        }
    renderImageToCanvas(clean, heroCleanCtx);
    renderImageToCanvas(diffusion.addNoise(clean, Math.floor(T * 0.45)), heroNoisyCtx);
    renderImageToCanvas(diffusion.addNoise(clean, T), heroPureCtx);
}
renderHeroPreview();

// ─── Disclosure toggles ───────────────────────────────────────────────────────
document.querySelectorAll('.disclosure-toggle').forEach(btn => {
    btn.addEventListener('click', () => {
        const body = btn.nextElementSibling as HTMLElement;
        const open = body.classList.toggle('open');
        btn.textContent = btn.textContent!.replace(/^[↓↑]/, open ? '↑' : '↓');
    });
});

// ─── Dataset panel ────────────────────────────────────────────────────────────
btnLoadPresets.addEventListener('click', () => {
    datasetPanel.loadPresets(SMILEY_PRESETS);
});

btnClearDataset.addEventListener('click', () => {
    datasetPanel.clearAll();
});

// ─── Training ─────────────────────────────────────────────────────────────────
btnTrain.addEventListener('click', async () => {
    if (btnTrain.dataset.state === 'running') {
        trainer.stopTraining();
        setBtnTrain('idle');
        btnGenerate.disabled = false;   // allow generation even if stopped early
        return;
    }

    const images = datasetPanel.getImages();
    if (images.length === 0) {
        alert('Draw at least one image or load the smiley presets first!');
        return;
    }

    setBtnTrain('running');
    lossChart.reset();

    const dbgX0Ctx    = initDebugCanvas('debug-x0');
    const dbgXtCtx    = initDebugCanvas('debug-xt');
    const dbgNoiseCtx = initDebugCanvas('debug-noise');

    await trainer.trainOnDataset(images, async (step, loss, dbg) => {
        epochValue.textContent  = step.toString();
        lossValueEl.textContent = loss.toFixed(5);
        lossChart.addPoint(loss);

        // Progress bar
        const pct = (step / CONFIG.trainingSteps) * 100;
        progressFill.style.width = `${pct}%`;
        progressLabel.textContent = `Step ${step} / ${CONFIG.trainingSteps}`;

        // Debug canvases
        if (dbg) {
            debugT.textContent = dbg.t.toString();
            renderImageToCanvas(dbg.x0, dbgX0Ctx);
            renderImageToCanvas(dbg.xt, dbgXtCtx);
            renderImageToCanvas(dbg.epsilon, dbgNoiseCtx);
        }
    });

    setBtnTrain('idle');
    btnGenerate.disabled = false;
});

function setBtnTrain(state: 'idle' | 'running') {
    btnTrain.dataset.state = state;
    btnTrain.textContent   = state === 'running' ? '⏹ Stop Training' : '▶ Start Training';
    btnTrain.className     = state === 'running' ? 'btn-sm' : 'btn-primary-sm';
}

btnSaveModel.addEventListener('click', async () => {
    const ok = await trainer.saveModel('diffusion-model');
    alert(ok ? 'Saved to browser storage + downloaded!' : 'Save failed. Check console.');
});

btnLoadModel.addEventListener('click', async () => {
    const ok = await trainer.loadModelFromIndexedDB('diffusion-model');
    if (ok) btnGenerate.disabled = false;
    alert(ok ? 'Model loaded from browser storage!' : 'No saved model found. Train first.');
});

fileLoadModel.addEventListener('change', async (e) => {
    const files = (e.target as HTMLInputElement).files;
    if (!files?.length) return;
    const ok = await trainer.loadModelFromFiles(files);
    if (ok) btnGenerate.disabled = false;
    alert(ok ? 'Model loaded from files!' : 'Failed — select both .json and .bin.');
    fileLoadModel.value = '';
});

// ─── Generation ────────────────────────────────────────────────────────────────
btnGenerate.addEventListener('click', async () => {
    btnGenerate.disabled  = true;
    btnGenerate.textContent = '⏳ Generating…';
    strip.clear();

    await trainer.generateImage((_t, data, progress) => {
        genProgressSlider.value      = progress.toString();
        genProgressValue.textContent = progress.toString();
        renderImageToCanvas(data, genIntermCtx);
        renderImageToCanvas(data, genCtx);
    }, 5);

    // Populate denoising strip with cached frames
    strip.load(trainer.lastFrames);

    btnGenerate.disabled  = false;
    btnGenerate.textContent = '✨ Generate';
});

// ─── Architecture section ─────────────────────────────────────────────────────

const forwardDemo = new ForwardProcessDemo(
    'forward-canvas', 'forward-slider', 'forward-t-label', 'forward-alpha-bar-label',
    'forward-play-btn', diffusion
);
forwardDemo.setBaseImage(SMILEY_PRESETS[0]);

(document.getElementById('forward-preset-select') as HTMLSelectElement).addEventListener('change', (e) => {
    const idx = parseInt((e.target as HTMLSelectElement).value);
    forwardDemo.setBaseImage(SMILEY_PRESETS[idx]);
});

new UNetExplorer('unet-canvas', 'unet-popup');

// ─── Robotics demo ────────────────────────────────────────────────────────────
const roboticsDemo = new RoboticsDemo('robotics-canvas');
btnRoboticsPlay.addEventListener('click', () => { roboticsDemo.play(); });

// Auto-animate when section scrolls into view
new IntersectionObserver((entries) => {
    if (entries[0].isIntersecting) roboticsDemo.play();
}, { threshold: 0.4 }).observe(document.getElementById('section-robotics')!);

// ─── Flow Matching demo ───────────────────────────────────────────────────────
const flowDemo = new FlowMatchingDemo('flow-canvas');
btnFlowPlay.addEventListener('click', () => { flowDemo.play(); });

new IntersectionObserver((entries) => {
    if (entries[0].isIntersecting) flowDemo.play();
}, { threshold: 0.4 }).observe(document.getElementById('section-flow')!);
