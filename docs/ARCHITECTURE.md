# Diffusion Explainer — Architecture Document

> Version 1.2 · März 2026 (aktualisiert: Sprint 2)
> Companion zu [REQUIREMENTS.md](./REQUIREMENTS.md)

---

## 1. System-Übersicht

```
┌────────────────────────────────────────────┐
│               PYTHON SIDE                  │
│  (offline, nur Dev-Maschine)               │
│                                            │
│  ml/                                       │
│  ├─ scripts/                               │
│  │   ├─ train_base_unet.py  ← ✅ DONE     │
│  │   └─ fix_model_json.py                  │
│  ├─ models/  (gitignored, .keras Dateien)  │
│  │   └─ base_unet.keras  ← ✅ DONE        │
│  ├─ requirements.txt                       │
│  └─ convert_to_tfjs.sh                     │
│       ↓ (tensorflowjs_converter)           │
└────────┬───────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  public/model/                      │
│  ├─ model.json   ← committed        │
│  └─ *.bin        ← gitignored       │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│                    BROWSER SIDE                         │
│  (Vite 7 + TypeScript strict + TensorFlow.js 4.22)     │
│                                                         │
│  src/                                                   │
│  ├─ core/                                               │
│  │   ├─ config.ts      CONFIG-Singleton                 │
│  │   ├─ diffusion.ts   DiffusionModel (Mathe)           │
│  │   └─ trainer.ts     ModelTrainer (U-Net, Loop, Gen)  │
│  ├─ components/                                         │
│  │   ├─ canvas.ts           Legacy DrawingCanvas        │
│  │   ├─ datasetPanel.ts     5-Slot Zeichenfläche        │
│  │   ├─ denoisingStrip.ts   Strip + Zoom-Panel          │
│  │   ├─ forwardDemo.ts      Forward-Process-Demo        │
│  │   ├─ lossChart.ts        Canvas-basierter Loss-Graph │
│  │   ├─ renderUtils.ts      Image/Canvas-Helpers        │
│  │   ├─ roboticsDemo.ts     Robotics + Flow Matching    │
│  │   └─ unetExplorer.ts     Klickbares U-Net-Diagramm  │
│  ├─ data/                                               │
│  │   └─ presets.ts     SMILEY_PRESETS, CLASS_PRESETS    │
│  ├─ main.ts         Slim App-Orchestrator               │
│  └─ style.css       Dark Amber Phosphor Theme           │
│                                                         │
│  index.html         Hero + 6 Sektionen + Footer         │
└─────────────────────────────────────────────────────────┘
```

**Kernprinzip:** Alles läuft nach dem initialen Page-Load vollständig im Browser. Python produziert nur pretrained Model-Assets während der Entwicklung.

---

## 2. Model-Architektur

### 2.1 Model A — Unconditional U-Net (Browser-trainiert)

**Status:** ✅ Implementiert (Sprint 1+2)

```
Zweck:    32×32 Graustufen-Bilder denoisen (gezeichnet oder Presets)
Training: Im Browser via TensorFlow.js (~3-4 Min auf GPU)
Inference: 400 DDPM-Schritte im Browser
Params:   ~118k (16/32/64 Filter — reduziert in Sprint 2)
```

**Architektur:**

```
Input A: x_t     [B, 32, 32, 1]
Input B: t_emb   [B, 16]          ← sinusoidales Embedding (neu Sprint 2)

Time-Input-Projektion:
  Dense(16→1024, kein relu)       ← kein relu: sin/cos hat [-1,1] Range
  Reshape(32, 32, 1)
  Concat([x_t, time_map])  → [B, 32, 32, 2]

Encoder:
  e1: Conv2D(16, 3×3) × 2  → [B, 32, 32, 16]
  p1: MaxPool(2×2)          → [B, 16, 16, 16]
  e2: Conv2D(32, 3×3) × 2  → [B, 16, 16, 32]
  p2: MaxPool(2×2)          → [B,  8,  8, 32]

Bottleneck:
  b:  Conv2D(64, 3×3) × 2  → [B, 8, 8, 64]

Bottleneck Time-Conditioning (neu Sprint 2):
  Dense(16→64, relu)
  Reshape(1, 1, 64)
  Add(b, time_bot)          → broadcast [B, 8, 8, 64]

Decoder (mit Skip Connections):
  u2: UpSample(2×2)         → [B, 16, 16, 64]
  cat2: Concat([u2, e2])    → [B, 16, 16, 96]
  d2: Conv2D(32, 3×3) × 2  → [B, 16, 16, 32]
  u1: UpSample(2×2)         → [B, 32, 32, 32]
  cat1: Concat([u1, e1])    → [B, 32, 32, 48]
  d1: Conv2D(16, 3×3)       → [B, 32, 32, 16]

Output:
  Conv2D(1, 1×1, linear)    → [B, 32, 32, 1]  ← ε̂ predicted noise
```

### 2.2 Sinusoidales Time-Embedding (ADR-001)

**Dimension:** 16 (8 sin + 8 cos Komponenten)

```typescript
// src/core/trainer.ts — sinEmb()
private sinEmb(tNorm: number, dim: number): Float32Array {
    const out    = new Float32Array(dim);
    const scaled = tNorm * 1000;
    for (let i = 0; i < dim / 2; i++) {
        const freq     = 1.0 / Math.pow(10000, (2 * i) / dim);
        out[2 * i]     = Math.sin(scaled * freq);
        out[2 * i + 1] = Math.cos(scaled * freq);
    }
    return out;
}
```

Warum: Scalar-Input → Dense(relu) konnte t=0 bis t=400 kaum unterscheiden (alle Neuronen hatten Breakpoints bei t≈0 durch HeNormal-Init). Sinusoidal gibt sofort reiches Frequenzspektrum.

### 2.3 Noise Schedule

```
β₁..β_T: linear von 0.0001 bis 0.02
ᾱ_t = ∏ᵢ₌₁ᵗ (1 - βᵢ)     (cumulatives Produkt der Alphas)

Forward:  x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε,  ε ~ N(0,I)
Reverse:  x_{t-1} = (x_t - (1-αt)/√(1-ᾱt) · ε̂) / √αt + σt · z
```

---

## 3. Scripts & Tooling

| Script | Sprache | Zweck | Status |
|--------|---------|-------|--------|
| `ml/scripts/train_base_unet.py` | Python | Pretrain U-Net auf synthetischen Formen (~6 Min CPU, 5000 Steps) | ✅ DONE |
| `ml/scripts/fix_model_json.py` | Python | Repariert model.json nach tfjs_converter | ✅ DONE |
| `ml/convert_to_tfjs.sh` | Bash | Keras → public/model/ (tensorflowjs_converter) | ✅ DONE |

**Python Environment:**
```bash
cd ml/
pip install -r requirements.txt  # tensorflow>=2.19, tensorflowjs>=4.22
python scripts/train_base_unet.py
bash convert_to_tfjs.sh
```

---

## 4. Datasets

### 4.1 SMILEY_PRESETS (8 Varianten)

Programmatisch generierte Float32Arrays (keine Netzwerk-Assets nötig):

| Index | Name | Beschreibung |
|-------|------|-------------|
| 0 | happy | Lächeln, große Augen |
| 1 | happy-wide | Breites Lächeln |
| 2 | happy-small | Kleines Lächeln |
| 3 | neutral | Gerader Mund |
| 4 | neutral-dots | Nur Augen |
| 5 | sad | Trauriger Mund |
| 6 | sad-frown | Tiefes Stirnrunzeln |
| 7 | surprised | Runder Mund |

Alle: Float32Array, Länge 1024 (32×32), Werte in [-1, 1].

### 4.2 CLASS_PRESETS (Sprint 3, geplant)

3 Klassen (happy/neutral/sad), je mehrere Varianten — für conditional Diffusion Training.

---

## 5. Website-Architektur

### 5.1 HTML-Struktur

```html
<nav>                          ← Sticky Nav mit Abschnitts-Links
<section id="section-hero">    ← Zwei-Spalten Hero
<section id="section-architecture">  ← Forward Demo + U-Net Explorer
<section id="section-train">   ← Dataset Panel + Training
<section id="section-generate"> ← Generation + Denoising Strip
<section id="section-conditioning">  ← Conditional (Sprint 3)
<section id="section-robotics"> ← Robotics Demo
<section id="section-flow">    ← Flow Matching Demo
<footer>
```

### 5.2 Komponenten-Übersicht

| Datei | Klasse | DOM-Element | Zweck |
|-------|--------|-------------|-------|
| `forwardDemo.ts` | `ForwardProcessDemo` | `#forward-demo` | Slider Clean↔Noise |
| `unetExplorer.ts` | `UNetExplorer` | `#unet-explorer` | Klickbares U-Net |
| `datasetPanel.ts` | `DatasetPanel` | `#dataset-panel` | 5 Zeichenslots |
| `lossChart.ts` | `LossChart` | `#loss-chart` | Canvas-Loss-Kurve |
| `denoisingStrip.ts` | `DenoisingStrip` | `#denoising-strip` | Thumbnails + Zoom |
| `roboticsDemo.ts` | `RoboticsDemo` | `#robotics-demo` | Pfadplanung-Animation |
| `roboticsDemo.ts` | `FlowMatchingDemo` | `#flow-demo` | DDPM vs. Flow |
| `renderUtils.ts` | — | — | Float32→Canvas Helpers |
| `canvas.ts` | `DrawingCanvas` | — | Legacy (compat) |

### 5.3 main.ts — App-Orchestrator

main.ts verdrahtet Komponenten mit DOM-Elementen und Training-Events. Kein eigener State. Slim bleibt Slim — max. ~150 Zeilen.

---

## 6. Test-Architektur

Siehe `docs/TEST_PLAN.md` für vollständige Test-Case-Dokumentation.

**Framework:** Vitest 4 + happy-dom
**Aktuell:** 4 Test-Dateien, 33 Tests, alle grün (~400ms)
**Kommando:** `npm test`

| Kategorie | Was wird getestet | Tools |
|-----------|-------------------|-------|
| Unit (pure Logik) | Config-Werte, Diffusion-Mathe, Presets | Vitest, kein DOM |
| Komponenten | Canvas-Rendering, Pixel-Konvertierung | Vitest + happy-dom + vi.fn() Mocks |
| Integration (geplant) | TF.js Modell-Training | Vitest + echtes TF.js |

---

## 7. Iterativer Build-Plan

| Version | Status | Was gebaut wurde |
|---------|--------|-----------------|
| v0.1 | ✅ DONE | Statische Erklärungsseite (ForwardDemo, U-Net Explorer) |
| v0.2 | ✅ DONE | Training + Generation |
| v0.3 | ✅ DONE | Denoising Strip + Zoom |
| v0.4 | 🔜 NEXT | Conditional Diffusion (Sprint 3) |
| v0.5 | ⬜ PLANNED | Robotics + Auto-Steering (Sprint 4) |
| v1.0 | ⬜ PLANNED | DDIM, Lazy Loading, GitHub Pages Deploy (Sprint 5) |

---

## 8. CI/CD

GitHub Actions via `.github/workflows/ci.yml`:

```
PR → TypeCheck (tsc --noEmit) → Tests (vitest run) → Build (vite build)
```

Branch Protection: PRs required, CI muss grün sein vor Merge.
