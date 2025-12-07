import './style.css';
import { DrawingCanvas } from './canvas';
import { CONFIG } from './config';
import { DiffusionModel } from './diffusion';

// Initialize Drawing Canvas
const drawingCanvas = new DrawingCanvas('drawing-canvas');

// Initialize Diffusion Model
const diffusionModel = new DiffusionModel();

// Initialize Noise Canvas (Output)
const noiseCanvas = document.getElementById('noise-canvas') as HTMLCanvasElement;
const noiseCtx = noiseCanvas.getContext('2d')!;
noiseCanvas.width = CONFIG.imageSize;
noiseCanvas.height = CONFIG.imageSize;
noiseCanvas.style.width = '256px';
noiseCanvas.style.height = '256px';
noiseCanvas.style.imageRendering = 'pixelated';

// UI Elements
const btnClear = document.getElementById('btn-clear') as HTMLButtonElement;
const fileUpload = document.getElementById('file-upload') as HTMLInputElement;
const noiseSlider = document.getElementById('noise-slider') as HTMLInputElement;
const noiseValue = document.getElementById('noise-value') as HTMLSpanElement;
const maxStepsValue = document.getElementById('max-steps') as HTMLSpanElement;
const btnAutoplay = document.getElementById('btn-autoplay') as HTMLButtonElement;

// Set Max Steps in UI
noiseSlider.max = diffusionModel.getTimesteps().toString();
maxStepsValue.textContent = diffusionModel.getTimesteps().toString();

// State
let currentT = 0;
let isPlaying = false;
let animationFrameId: number | null = null;

// Event Listeners
btnClear.addEventListener('click', () => {
  drawingCanvas.clear();
  updateNoisyImage();
});

fileUpload.addEventListener('change', (e) => {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (!file) return;

  const img = new Image();
  img.onload = () => {
    drawingCanvas.drawImage(img);
    updateNoisyImage();
  };
  img.src = URL.createObjectURL(file);
});

noiseSlider.addEventListener('input', (e) => {
  if (isPlaying) stopAutoplay();
  const val = parseInt((e.target as HTMLInputElement).value);
  currentT = val;
  updateUI();
  updateNoisyImage();
});

btnAutoplay.addEventListener('click', () => {
  if (isPlaying) {
    stopAutoplay();
  } else {
    startAutoplay();
  }
});

// Also update when drawing ends to keep the noisy view in sync
document.getElementById('drawing-canvas')?.addEventListener('mouseup', updateNoisyImage);
document.getElementById('drawing-canvas')?.addEventListener('touchend', updateNoisyImage);
document.getElementById('drawing-canvas')?.addEventListener('mouseout', updateNoisyImage);

function updateUI() {
  noiseSlider.value = currentT.toString();
  noiseValue.textContent = currentT.toString();
}

function startAutoplay() {
  isPlaying = true;
  btnAutoplay.textContent = '⏸ Pause';

  // Reset if at end
  if (currentT >= diffusionModel.getTimesteps()) {
    currentT = 0;
  }

  const totalSteps = diffusionModel.getTimesteps();
  // Target ~5 seconds for the full animation
  // 60 FPS = 16ms per frame. 5000ms / 16ms ≈ 300 frames.
  // Steps per frame = Total Steps / 300
  const stepIncrement = Math.max(1, Math.ceil(totalSteps / 300));

  const animate = () => {
    if (!isPlaying) return;

    if (currentT < totalSteps) {
      currentT = Math.min(totalSteps, currentT + stepIncrement);
      updateUI();
      updateNoisyImage();
      animationFrameId = requestAnimationFrame(animate);
    } else {
      stopAutoplay();
    }
  };
  animate();
}

function stopAutoplay() {
  isPlaying = false;
  btnAutoplay.textContent = '▶ Auto-Play';
  if (animationFrameId) cancelAnimationFrame(animationFrameId);
}

// Core Logic: Add Noise using Diffusion Model
function updateNoisyImage() {
  const originalImageData = drawingCanvas.getImageData();
  const pixels = originalImageData.data;

  // Convert ImageData (Uint8ClampedArray) to Float32Array [-1, 1] for model
  const x0 = new Float32Array(CONFIG.imageSize * CONFIG.imageSize);
  for (let i = 0; i < x0.length; i++) {
    // Normalize 0..255 -> -1..1
    x0[i] = (pixels[i * 4] / 127.5) - 1.0;
  }

  // Apply Diffusion Forward Process
  const xt = diffusionModel.addNoise(x0, currentT);

  // Convert back to ImageData for display
  const noisyImageData = noiseCtx.createImageData(CONFIG.imageSize, CONFIG.imageSize);
  const output = noisyImageData.data;

  for (let i = 0; i < xt.length; i++) {
    // Denormalize -1..1 -> 0..255
    let val = (xt[i] + 1.0) * 127.5;
    val = Math.max(0, Math.min(255, val)); // Clip

    output[i * 4] = val;     // R
    output[i * 4 + 1] = val; // G
    output[i * 4 + 2] = val; // B
    output[i * 4 + 3] = 255; // Alpha
  }

  noiseCtx.putImageData(noisyImageData, 0, 0);
}

// Initial render
updateNoisyImage();
