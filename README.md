# Diffusion Explainer

An interactive browser-based learning platform that teaches how Diffusion Models work — from the math to real-world robotics applications.

**Inspired by** [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) and [Teachable Machine](https://teachablemachine.withgoogle.com/).

> 100% in-browser. No backend. No login. No install.

---

## What you can do

- **Understand** the Forward Process (adding noise) interactively with a slider
- **Explore** the U-Net architecture by clicking through encoder/bottleneck/decoder blocks
- **Train** your own diffusion model on drawings you make in the browser
- **Watch** every denoising step and zoom in to see what the model predicts
- **See** how the same algorithm powers robot motion planning

---

## Quick Start

```bash
# Prerequisites: Node.js 22+
git clone https://github.com/zornalex/diffusion_explorer.git
cd diffusion_explorer
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173)

---

## Commands

```bash
npm run dev      # Dev server with hot reload
npm run build    # Production build → dist/
npm run preview  # Preview production build
npm test         # Run all tests (33 unit tests, ~400ms)
npx tsc --noEmit # TypeScript type check
```

---

## Project Structure

```
src/
  core/           – Diffusion model logic (config, math, training)
  components/     – UI components (all canvas-based, no framework)
  data/           – Programmatic datasets (smiley presets)
  main.ts         – App wiring
  style.css       – Dark Amber Phosphor theme
ml/
  scripts/        – Python pre-training scripts (offline dev only)
  convert_to_tfjs.sh – Keras → TF.js conversion
public/model/     – Pre-trained TF.js weights (model.json + *.bin)
docs/
  ARCHITECTURE.md – Technical architecture details
  REQUIREMENTS.md – Product requirements and user stories
  TEST_PLAN.md    – Complete test case documentation
```

---

## Python ML Pipeline (optional, dev only)

The browser uses a pre-trained base model. To retrain it:

```bash
cd ml/
pip install -r requirements.txt   # tensorflow>=2.19, tensorflowjs>=4.22
python scripts/train_base_unet.py  # ~6 min on CPU
bash convert_to_tfjs.sh            # → public/model/
```

The Python model architecture must match the TypeScript U-Net exactly. See `docs/ARCHITECTURE.md` for details.

---

## Tech Stack

- **Frontend:** Vite 7 + TypeScript (strict), no framework
- **ML:** TensorFlow.js 4.22 (in-browser training + inference)
- **Testing:** Vitest 4 + happy-dom
- **CI:** GitHub Actions (typecheck + test + build on every PR)
- **Python Pipeline:** TensorFlow 2.19 + tensorflowjs 4.22

---

## Based on

- Ho et al. (2020) — [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- Chi et al. (2023) — [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)
- Lipman et al. (2022) — [Flow Matching](https://arxiv.org/abs/2210.02747)
