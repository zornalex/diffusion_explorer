import './style.css';
import { DrawingCanvas } from './canvas';
import { CONFIG } from './config';

// Initialize Drawing Canvas
const drawingCanvas = new DrawingCanvas('drawing-canvas');

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

// State
let currentNoiseLevel = 0;

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
  const val = parseInt((e.target as HTMLInputElement).value);
  currentNoiseLevel = val / 100;
  noiseValue.textContent = `${val}%`;
  updateNoisyImage();
});

// Also update when drawing ends to keep the noisy view in sync
document.getElementById('drawing-canvas')?.addEventListener('mouseup', updateNoisyImage);
document.getElementById('drawing-canvas')?.addEventListener('touchend', updateNoisyImage);
document.getElementById('drawing-canvas')?.addEventListener('mouseout', updateNoisyImage);

// Core Logic: Add Noise (Simple linear interpolation for PoC)
function updateNoisyImage() {
  const originalData = drawingCanvas.getImageData();
  const noisyData = noiseCtx.createImageData(CONFIG.imageSize, CONFIG.imageSize);

  const pixels = originalData.data;
  const output = noisyData.data;

  for (let i = 0; i < pixels.length; i += 4) {
    // Get grayscale value (r=g=b in our case)
    const pixelVal = pixels[i] / 255; // 0..1

    // Generate random noise
    const noise = Math.random(); // 0..1

    // Linear interpolation: (1 - alpha) * image + alpha * noise
    // This isn't the exact diffusion formula yet, but good for visual PoC
    const alpha = currentNoiseLevel;
    const noisyVal = (1 - alpha) * pixelVal + alpha * noise;

    // Write back to output
    const finalVal = Math.floor(noisyVal * 255);
    output[i] = finalVal;     // R
    output[i + 1] = finalVal; // G
    output[i + 2] = finalVal; // B
    output[i + 3] = 255;      // Alpha
  }

  noiseCtx.putImageData(noisyData, 0, 0);
}

// Initial render
updateNoisyImage();
