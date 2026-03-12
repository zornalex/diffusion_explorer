# Docs Reorganisation + CI/CD Setup — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Saubere Dateistruktur (Altlasten löschen, docs/ Ordner), vollständige Dokumentation (CLAUDE.md, README.md, TEST_PLAN.md), und GitHub Actions CI/CD Pipeline (Tests + TypeCheck + Build bei jedem PR).

**Architecture:** Keine Code-Änderungen — nur Dateiorganisation und neue Markdown-Dokumente. GitHub Actions `.github/workflows/ci.yml` als einzige neue "Infra"-Datei. Branch Protection muss manuell im GitHub UI aktiviert werden.

**Tech Stack:** Git, GitHub Actions, Node.js 22, Vitest 4, TypeScript strict (`tsc --noEmit`), Vite 7

---

## Chunk 1: Cleanup + Dokumentation

### Task 1: Altlasten löschen + .gitignore aktualisieren

**Files:**
- Delete: `unet.onnx` (tracked im Repo)
- Delete: `unet.onnx.data` (tracked im Repo)
- Delete: `unet_opt.onnx` (tracked im Repo)
- Delete: `convert_flax_to_tf.py` (untracked)
- Delete: `convert_model.py` (untracked)
- Delete: `optimize_onnx.py` (untracked)
- Delete: `try_tf_loading.py` (untracked)
- Modify: `.gitignore` — `model/` Root-Verzeichnis hinzufügen

- [ ] **Step 1: Tracked ONNX-Dateien via git rm entfernen**

```bash
cd /Users/azorn/dev/diffusion_explorer
# Prüfen ob ONNX-Dateien getrackt sind
git ls-files unet.onnx unet.onnx.data unet_opt.onnx
```

Expected: Gibt die Dateinamen aus (= tracked). Falls leer → stattdessen `rm` verwenden.

```bash
git rm unet.onnx unet.onnx.data unet_opt.onnx
```

- [ ] **Step 2: Untracked Python-Scripts löschen**

```bash
rm convert_flax_to_tf.py convert_model.py optimize_onnx.py try_tf_loading.py
```

- [ ] **Step 3: `model/` Root-Verzeichnis zu .gitignore hinzufügen**

Das `model/` Verzeichnis enthält `mnist_diffusion_unet_1500.keras` (große .keras Datei, darf nicht committed werden).

Füge am Ende von `.gitignore` hinzu:
```
# Root-level experiment models (large .keras, not for repo)
model/
```

- [ ] **Step 4: Verify**

```bash
ls *.onnx 2>/dev/null || echo "OK — no onnx files"
ls *.py 2>/dev/null || echo "OK — no py files at root"
git status --short | grep "model/" || echo "OK — model/ gitignored"
```

Expected: alle drei Checks "OK"

---

### Task 2: docs/ Verzeichnis + REQUIREMENTS.md

**Files:**
- Create: `docs/REQUIREMENTS.md` (verschoben + aktualisiert)
- Delete: `REQUIREMENTS.md` (root)

- [ ] **Step 1: docs/ Verzeichnis anlegen**

```bash
mkdir -p docs
```

- [ ] **Step 2: REQUIREMENTS.md nach docs/ verschieben und aktualisieren**

Erstelle `docs/REQUIREMENTS.md` mit folgendem Inhalt (Sprint-Status aktualisiert):

```markdown
# Diffusion Explainer — Product Requirements & User Stories

> Version 3.1 · März 2026 (aktualisiert)
> Design-Vorbilder: [Transformer Explainer](https://poloclub.github.io/transformer-explainer/), [Teachable Machine](https://teachablemachine.withgoogle.com/)

---

## 1. Vision

Eine interaktive, 100% browser-basierte Lernplattform, die Teilnehmern vermittelt, wie Diffusion Models funktionieren — von den Grundlagen bis zur Anwendung in der Robotik.

**Kernprinzip:** Hands-on first. Jeder Nutzer soll die Mechanismen selbst erleben, bevor Erklärungen folgen. Kein Backend, kein Login, kein Install.

**Drei Säulen:**

1. **Verstehen** — Durch die Architektur eines Diffusion Models klicken und jeden Schritt visuell nachvollziehen
2. **Selbst machen** — Bilder zeichnen, ein eigenes Modell im Browser trainieren (Teachable Machine-Prinzip)
3. **Anwenden** — Sehen, wie dasselbe Prinzip in der Robotik für Trajektorien eingesetzt wird

---

## 2. Sprint-Übersicht

| Sprint | Status | Thema |
|--------|--------|-------|
| Sprint 1 | ✅ DONE | Dataset-Panel, Training, Denoising Strip, Zoom-Panel, Presets, Robotics |
| Sprint 2 | ✅ DONE | Sinusoidales Embedding, GPU-Loop, ForwardDemo, U-Net Explorer, Frontend-Redesign |
| Sprint 3 | 🔜 NEXT | Conditional Diffusion (Klassen-Labels, CFG) |
| Sprint 4 | ⬜ PLANNED | Auto-Steering Demo (deterministisch) |
| Sprint 5 | ⬜ PLANNED | DDIM, GitHub Pages Deploy, Polish |

---

## 3. User Stories

### Epic 1: Architektur verstehen

| ID | User Story | Status | Akzeptanzkriterien |
|---|---|---|---|
| **US-1.1** | Als TN will ich sehen, wie der Forward Process funktioniert. | ✅ DONE | ForwardProcessDemo: Slider Clean→Noise, festes ε pro Bild (Box-Muller), Preset-Auswahl |
| **US-1.2** | Als TN will ich die U-Net Architektur interaktiv erkunden. | ✅ DONE | UNetExplorer: 7 klickbare Blöcke (Input/Encoder/Bottleneck/Decoder/Output), Detail-Panel |
| **US-1.3** | Als TN will ich den Reverse Process Schritt für Schritt sehen. | ✅ DONE | DenoisingStrip: ~20 Thumbnails, Klick → Zoom-Panel [xₜ → ε̂ → xₜ₋₁] |
| **US-1.4** | Als TN will ich selbst entscheiden wie tief ich einsteige. | ✅ DONE | Progressive Disclosure: aufklappbare Sections (Formeln, Pseudocode) |

### Epic 2: Selbst trainieren

| ID | User Story | Status | Akzeptanzkriterien |
|---|---|---|---|
| **US-2.1** | Als TN will ich eigene Bilder zeichnen oder Presets laden. | ✅ DONE | DatasetPanel: 5 Slots, Maus+Touch, "Load Smileys"-Button (8 programmatische Presets) |
| **US-2.2** | Als TN will ich live sehen wie das Modell lernt. | ✅ DONE | Training: Loss-Chart, Fortschrittsbalken, Milestone-Previews (Step 200/800/2000), Stop-Button |
| **US-2.3** | Als TN will ich Bilder generieren und Denoising nachvollziehen. | ✅ DONE | Generate: DDPM Reverse (400 Steps), DenoisingStrip + Zoom-Panel |
| **US-2.4** | Als TN will ich mein Modell speichern und laden. | ✅ DONE | Save/Load via TF.js model.save() / model.loadLayersModel() |

### Epic 3: Conditional Diffusion Models

| ID | User Story | Status | Akzeptanzkriterien |
|---|---|---|---|
| **US-3.1** | Als TN will ich Klassen-Konditionierung verstehen. | 🔜 Sprint 3 | Erklärung unconditioned vs. conditioned, CFG-Slider |
| **US-3.2** | Als TN will ich ein conditional Modell selbst trainieren. | 🔜 Sprint 3 | 3-Klassen-UI (😊/😐/😢), Klasse auswählen → generieren |
| **US-3.3** | Als TN will ich ein vortrainiertes conditional Modell nutzen. | 🔜 Sprint 3 | Pretrained model in public/model/conditional-smiley/ |
| **US-3.4** | Als TN will ich den Bezug zu Text-zu-Bild-Modellen verstehen. | 🔜 Sprint 3 | Erklärung CLIP-Vektor als hochdim. Klassen-Label |

### Epic 4: Robotik — Flow Matching

| ID | User Story | Status | Akzeptanzkriterien |
|---|---|---|---|
| **US-4.1** | Als TN will ich Flow Matching vs. DDPM verstehen. | ✅ DONE | FlowMatchingDemo: Animation DDPM (Zickzack) vs. Flow Matching (gerader Pfad) |
| **US-4.2** | Als TN will ich Diffusion in der Robotik sehen. | ✅ DONE | Erklärung Pixel→Actions, RoboticsDemo |
| **US-4.3** | Als TN will ich eine Live-Demo mit Pfadgenerierung sehen. | ✅ DONE | RoboticsDemo: 2D Canvas, Hindernisse, generierter Pfad, Animation |
| **US-4.5** | Relevante Paper verlinkt. | ✅ DONE | Links zu DDPM, Diffusion Policy, Flow Matching |

---

## 4. Progressive Disclosure

| Layer | Was wird gezeigt | Sichtbarkeit |
|---|---|---|
| **Layer 0 — Visuell** | Animationen, Canvases, Slider | Immer sichtbar |
| **Layer 1 — Intuitiv** | 1-3 Sätze, Analogien | Standard-Text |
| **Layer 2 — Konzeptuell** | Pseudocode, Schritt-Erklärung | Aufklappbar |
| **Layer 3 — Mathematisch** | DDPM-Formeln, Paper-Links | Aufklappbar |

---

## 5. Technische Constraints

- **Kein Backend** — alles läuft im Browser nach initialem Page-Load
- **Modellgröße** — max ~3 MB für pretrained weights (GitHub Pages Limit beachten)
- **Browser-Kompatibilität** — Chrome/Firefox mit WebGL für TF.js GPU-Beschleunigung
- **Bildgröße** — 32×32 Pixel (intern), dargestellt 80-128px für Lesbarkeit
```

- [ ] **Step 3: Altes REQUIREMENTS.md löschen**

```bash
rm REQUIREMENTS.md
```

---

### Task 3: ARCHITECTURE.md nach docs/ verschieben und aktualisieren

**Files:**
- Create: `docs/ARCHITECTURE.md` (verschoben + aktualisiert)
- Delete: `ARCHITECTURE.md` (root)

- [ ] **Step 1: docs/ARCHITECTURE.md schreiben**

Erstelle `docs/ARCHITECTURE.md`:

```markdown
# Diffusion Explainer — Architecture Document

> Version 1.1 · März 2026 (aktualisiert: Sprint 2)
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
│  │   ├─ diffusion.ts   DiffusionModel (Mathe)          │
│  │   └─ trainer.ts     ModelTrainer (U-Net, Loop, Gen) │
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
│  │   └─ presets.ts     SMILEY_PRESETS, CLASS_PRESETS   │
│  ├─ main.ts         Slim App-Orchestrator               │
│  └─ style.css       Dark Amber Phosphor Theme           │
│                                                         │
│  index.html         6 Sektionen + Hero + Footer         │
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

### 2.3 Noise Schedule (Cosine / Linear Beta)

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

### Kategorien

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
```

- [ ] **Step 2: Altes ARCHITECTURE.md löschen**

```bash
rm ARCHITECTURE.md
```

---

### Task 4: docs/TEST_PLAN.md erstellen

**Files:**
- Create: `docs/TEST_PLAN.md`

- [ ] **Step 1: docs/TEST_PLAN.md schreiben**

```markdown
# Diffusion Explainer — Test Plan

> Stand: März 2026 · 33 Tests · 4 Test-Dateien · alle grün

---

## Wie man Tests ausführt

```bash
npm test              # Alle Tests einmal ausführen (~400ms)
npm run test:watch    # Watch-Modus für TDD
```

Erwartete Ausgabe:
```
✓ src/core/config.test.ts        (7 tests)
✓ src/components/renderUtils.test.ts (5 tests)
✓ src/core/diffusion.test.ts    (11 tests)
✓ src/data/presets.test.ts      (10 tests)

Test Files  4 passed (4)
      Tests 33 passed (33)
   Duration ~400ms
```

---

## Test-Kategorien

| Kategorie | Dateien | Tests | Beschreibung |
|-----------|---------|-------|-------------|
| **Unit — Config** | `config.test.ts` | 7 | Konfigurationswerte und Constraints |
| **Unit — Diffusion** | `diffusion.test.ts` | 11 | Noise-Schedule-Mathematik, Forward-Process |
| **Unit — Presets** | `presets.test.ts` | 10 | Datensatz-Integrität, Bildformate |
| **Komponente — Render** | `renderUtils.test.ts` | 5 | Pixel-Konvertierung Float32↔Uint8 |

---

## src/core/config.test.ts (7 Tests)

Testet den `CONFIG`-Singleton aus `src/core/config.ts`. Stellt sicher dass keine versehentlichen Wertänderungen unbemerkt bleiben.

| # | Test-Name | Was wird geprüft | Warum wichtig |
|---|-----------|-----------------|----------------|
| 1 | `has imageSize of 32` | `CONFIG.imageSize === 32` | Alle Tensor-Shapes, Canvas-Größen und SMILEY_PRESETS setzen exakt 32×32 voraus. Eine Änderung würde das gesamte System brechen. |
| 2 | `has timesteps of 400` | `CONFIG.timesteps === 400` | DDPM-Forward- und Reverse-Prozess laufen genau T=400 Schritte. Noise-Schedule-Arrays haben Länge 400. |
| 3 | `has betaStart less than betaEnd` | `CONFIG.betaStart < CONFIG.betaEnd` | Noise-Schedule muss monoton ansteigen: wenig Rauschen bei t=1, viel bei t=T. Vertauschen würde den Forward Process umkehren. |
| 4 | `has trainingSteps of 2000 (matching milestone labels)` | `CONFIG.trainingSteps === 2000` | Milestone-Trigger in main.ts prüfen auf Step 200/800/2000. Andere Werte würden Milestone-Previews nie oder falsch auslösen. |
| 5 | `has timeDim of 16 (sinusoidal embedding dimension)` | `CONFIG.timeDim === 16` | Das U-Net erwartet `[B, 16]` als Zeit-Input. Python-Pretraining-Script muss identisch sein. Wertänderung → Modell-Incompatibility. |
| 6 | `has positive learning rate` | `0 < CONFIG.learningRate < 1` | LR=0 → kein Training; LR≥1 → Gradient Explosion. Plausibilitätscheck. |
| 7 | `has positive batchSize` | `CONFIG.batchSize > 0` | BatchSize=0 würde leere Tensoren erzeugen und TF.js-Fehler produzieren. |

---

## src/core/diffusion.test.ts (11 Tests)

Testet `DiffusionModel` aus `src/core/diffusion.ts` — die mathematische Kernlogik des DDPM Forward Process.

| # | Test-Name | Was wird geprüft | Warum wichtig |
|---|-----------|-----------------|----------------|
| 1 | `returns correct timestep count` | `model.getTimesteps() === 400` | Konsistenz zwischen DiffusionModel und CONFIG. |
| 2 | `alphaBar at t<0 returns 1 (boundary)` | `model.getAlphaBar(-1) === 1` | Boundary-Handling: t<0 ist kein gültiger Zeitschritt. αbar=1 bedeutet kein Rauschen — sicheres Fallback. |
| 3 | `alphaBar is close to 1 at t=1 (first step)` | `0.98 < getAlphaBar(0) ≤ 1` | Bei t=1 (erster Schritt) soll fast kein Rauschen addiert werden. Stellt sicher dass der Noise-Schedule sanft beginnt. |
| 4 | `alphaBar is close to 0 at t=T (full noise)` | `0 ≤ getAlphaBar(399) < 0.05` | Bei t=T soll das Bild fast vollständiges Rauschen sein (αbar≈0). Voraussetzung für Generation aus reinem Rauschen. |
| 5 | `alphaBar is monotonically decreasing` | `getAlphaBar(t) < getAlphaBar(t-1)` für alle t | Noise-Schedule muss streng monoton fallen. Nicht-monotone Schedule würde den Reverse-Process mathematisch inkonsistent machen. |
| 6 | `getAlpha at t<0 returns 1` | `model.getAlpha(-1) === 1` | Boundary-Handling für α (nicht ᾱ). Kein Rauschen bei ungültigem Index. |
| 7 | `all getAlpha values are in (0, 1)` | `0 < getAlpha(t) < 1` für alle t | αt = 1 - βt. Muss strikt in (0,1) liegen: αt=1 → kein Rauschen; αt≤0 → mathematisch undefiniert. |
| 8 | `addNoise at t=0 returns original image unchanged` | `addNoise(img, 0) ≈ img` | Bei t=0: ᾱ₀=1, also x₀ = 1·x₀ + 0·ε = x₀. Wichtig für den ForwardProcessDemo-Slider am linken Anschlag. |
| 9 | `addNoiseWithEpsilon is deterministic given same epsilon` | Zweimal aufgerufen mit gleichem ε → gleiches Ergebnis | `addNoiseWithEpsilon` nimmt festes ε (Box-Muller, einmal generiert pro Bild). Determinismus ist Kern des ForwardDemo-Features: gleicher Slider-Wert zeigt immer dasselbe Bild. |
| 10 | `addNoiseWithEpsilon at t=0 returns original` | `addNoiseWithEpsilon(img, 0, eps) = img` | Spezialfall: bei t=0 kein ε-Anteil. Kompatibel mit Test #8. |
| 11 | `addNoise at high t has high variance (more noise than signal)` | `mean(addNoise(all-white, T-1)) < 0.7` | Stochastischer Smoke-Test: Bei t=T-1 muss das Signal fast verschwunden sein. Wenn das Bild noch weiß wäre, ist der Noise-Schedule kaputt. |

---

## src/data/presets.test.ts (10 Tests)

Testet `SMILEY_PRESETS`, `SMILEY_LABELS`, `CLASS_PRESETS`, `CLASS_LABELS` aus `src/data/presets.ts`.

| # | Test-Name | Was wird geprüft | Warum wichtig |
|---|-----------|-----------------|----------------|
| 1 | `contains 8 presets` | `SMILEY_PRESETS.length === 8` | DatasetPanel hat 5 Slots, lädt maximal 5 Presets. 8 Presets stellen sicher dass genug Varianten da sind (happy/neutral/sad je mehrere). |
| 2 | `each preset is a Float32Array` | `preset instanceof Float32Array` für alle | Das gesamte System erwartet Float32Array als Bildformat. Würde jemand ein Uint8Array übergeben, wären alle Pixel-Werte falsch (0-255 statt -1..1). |
| 3 | `each preset has size imageSize*imageSize` | `preset.length === 1024` für alle | U-Net-Input muss exakt 32×32=1024 Werte haben. Falsche Länge → Tensor-Reshape-Error in TF.js. |
| 4 | `all pixel values are in [-1, 1]` | `-1 ≤ v ≤ 1` für jeden Pixelwert | TF.js-Training und Rendering erwarten normalisierte Werte. Werte außerhalb führen zu falschem Training oder Clipping beim Render. |
| 5 | `different smiley variants are not identical (happy vs sad)` | `sum(|happy - sad|) > 0` | Stellt sicher dass die Presets tatsächlich unterschiedliche Bilder sind. Gleiche Bilder → Modell kann keine Distribution lernen. |
| 6 | `SMILEY_LABELS has same length as SMILEY_PRESETS` | `SMILEY_LABELS.length === 8` | Labels werden parallel zu Presets verwendet (UI-Anzeige). Unterschiedliche Längen → Array-Out-of-Bounds. |
| 7 | `CLASS_PRESETS has 3 classes (happy, neutral, sad)` | `CLASS_PRESETS.length === 3` | Conditional Diffusion (Sprint 3) erwartet genau 3 Klassen (Index 0=happy, 1=neutral, 2=sad). |
| 8 | `each class has at least one image` | `CLASS_PRESETS[i].length > 0` für alle i | Leere Klasse → Training kann keine Samples aus dieser Klasse ziehen → Gradient-Fehler. |
| 9 | `each class image is Float32Array of correct size` | Float32Array + length===1024 für alle Klassenbilder | Gleiche Constraints wie bei SMILEY_PRESETS. Konistenz zwischen unconditional und conditional Dataset. |
| 10 | `CLASS_LABELS has same length as CLASS_PRESETS` | `CLASS_LABELS.length === 3` | Analog zu Test #6 für Klassen-Labels. |

---

## src/components/renderUtils.test.ts (5 Tests)

Testet `renderImageToCanvas()` aus `src/components/renderUtils.ts` — die einzige Stelle im System wo Float32Array[-1,1] in Uint8[0,255]-Pixel umgewandelt wird.

Canvas und Context werden als Mocks simuliert (vi.fn()), da happy-dom kein echtes Canvas-Rendering unterstützt.

| # | Test-Name | Was wird geprüft | Warum wichtig |
|---|-----------|-----------------|----------------|
| 1 | `maps input -1.0 to pixel value 0` | `-1.0 → R=0, A=255` | Schwarzwert: -1 (Min des Normalisierungsbereichs) muss auf Schwarz (0) gemappt werden. Falsch: Bilder erscheinen verkehrt herum (Rauschen als weiß). |
| 2 | `maps input 1.0 to pixel value 255` | `1.0 → R=255, A=255` | Weißwert: 1 (Max) muss auf 255 gemappt werden. Zusammen mit Test #1 verifiziert das die lineare Skalierung `(v+1)/2*255`. |
| 3 | `maps input 0.0 to pixel value approximately 127-128` | `0.0 → R ∈ [127,128]` | Mittelwert: 0 soll Mittelgrau ergeben. Floating-Point-Rounding erlaubt ±1. Prüft dass die Formel symmetrisch ist. |
| 4 | `always sets alpha channel to 255` | `A === 255` für beliebigen Input | Alpha muss immer opaque sein. Transparentes Canvas ist in den Denoising-Strip-Thumbnails und im Zoom-Panel nicht sichtbar. |
| 5 | `clips values outside [-1, 1] gracefully` | `5.0 → R=255` (kein Crash, kein Overflow) | TF.js kann bei schlecht trainiertem Modell Werte außerhalb [-1,1] ausgeben. Clipping verhindert Uint8-Overflow (>255 würde zu falschen Farben wrappen). |

---

## Coverage-Gaps (noch nicht getestet)

| Datei | Was fehlt | Priorität |
|-------|-----------|-----------|
| `src/core/trainer.ts` | `sinEmb()` Ausgabe-Werte, `createModel()` Output-Shape, Training-Loop Konvergenz | 🔴 Hoch |
| `src/components/datasetPanel.ts` | Slot-Management, Zeichnen, getImages() | 🟡 Mittel |
| `src/components/denoisingStrip.ts` | Frame-Rendering, Zoom-Panel-Öffnen | 🟡 Mittel |
| `src/components/forwardDemo.ts` | Slider-Interaktion, Preset-Switching | 🟡 Mittel |
| `src/components/unetExplorer.ts` | Block-Klick-Events, Detail-Panel-Inhalt | 🟡 Mittel |
| `src/main.ts` | App-Initialisierung, Event-Verdrahtung | 🟢 Niedrig |

**Sprint 3 Test-Ziele:** trainer.ts (sinEmb, createModel, Training-Loop) priorisieren — das ist die kritischste untestete Logik.

---

## Test-Konventionen

```typescript
// ✅ So: describe + it, Datei neben Source
// src/core/foo.ts → src/core/foo.test.ts
describe('FooClass', () => {
    it('does specific thing with specific input', () => {
        expect(result).toBe(expected);
    });
});

// ❌ Nicht: bare test(), separater tests/ Ordner
// ❌ Nicht: setTimeout in Tests
// ❌ Nicht: TF.js Model-Methoden mocken — teste stattdessen pure Mathe-Funktionen
// ❌ Nicht: DOM-Tests ohne happy-dom-Environment (in vitest.config.ts konfiguriert)
```
```

---

### Task 5: README.md erstellen

**Files:**
- Create: `README.md`

- [ ] **Step 1: README.md schreiben**

```markdown
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
- **Python Pipeline:** TensorFlow 2.19 + tensorflowjs 4.22

---

## Based on

- Ho et al. (2020) — [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- Chi et al. (2023) — [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)
- Lipman et al. (2022) — [Flow Matching](https://arxiv.org/abs/2210.02747)
```

---

### Task 6: CLAUDE.md erstellen

**Files:**
- Create: `CLAUDE.md`

- [ ] **Step 1: CLAUDE.md schreiben**

```markdown
# Diffusion Explainer — CLAUDE.md

Interactive browser-based platform teaching Diffusion Models. 100% in-browser (TF.js), no backend. Modeled after Transformer Explainer + Teachable Machine.

Repo: https://github.com/zornalex/diffusion_explorer

---

## Tech Stack & Commands

- **Frontend:** Vite 7 + TypeScript strict (no framework), TensorFlow.js 4.22
- **Testing:** Vitest 4 + happy-dom
- **Python Pipeline:** TensorFlow 2.19 + tensorflowjs 4.22 (offline only, dev machine)

```bash
npm run dev          # Dev server → http://localhost:5173
npm run build        # tsc + vite build → dist/
npm test             # Vitest — 33 Tests, ~400ms
npm run test:watch   # TDD Watch-Modus
npx tsc --noEmit     # TypeCheck (= Lint in CI)
```

**Python Pipeline** (nur wenn Modell neu trainiert werden muss):
```bash
cd ml/
pip install -r requirements.txt
python scripts/train_base_unet.py   # ~6 Min CPU, → ml/models/base_unet.keras
bash convert_to_tfjs.sh             # → public/model/model.json + *.bin
```

---

## Architecture Quick-Reference

Vollständige Doku: `docs/ARCHITECTURE.md`

```
src/core/config.ts       CONFIG-Singleton
                         imageSize=32, timesteps=400, timeDim=16
                         batchSize=8, trainingSteps=2000, lr=0.001

src/core/diffusion.ts    DiffusionModel
                         Cosine Noise-Schedule, getAlphaBar(t), addNoise()
                         addNoiseWithEpsilon() für deterministische ForwardDemo

src/core/trainer.ts      ModelTrainer
                         createModel() — U-Net mit sinusoidalem Time-Embedding
                         sinEmb(tNorm, dim) — dim=16, Transformer-style
                         trainOnDataset() — GPU-only Loop, tf.tidy()
                         generateImage() — DDPM Reverse, cached Frames

src/components/          UI-Komponenten (alle Canvas-basiert)
  datasetPanel.ts        5 Zeichenslots, Load Smileys Button
  denoisingStrip.ts      ~20 Thumbnails + Zoom-Panel [xₜ→ε̂→xₜ₋₁]
  forwardDemo.ts         Slider Clean↔Noise, fixes ε pro Bild
  unetExplorer.ts        7 klickbare Blöcke (Encoder/Bottleneck/Decoder)
  lossChart.ts           Canvas Loss-Kurve
  roboticsDemo.ts        Robotics + Flow Matching Animationen
  renderUtils.ts         renderImageToCanvas() — einzige Float32→Pixel Stelle

src/data/presets.ts      SMILEY_PRESETS (8 programmatische 32×32 Float32Arrays)
                         CLASS_PRESETS (3 Klassen, Sprint 3)

src/main.ts              Slim Orchestrator — verdrahtet Komponenten mit DOM
                         Max. ~150 Zeilen. Kein eigener State.
```

**Model Input/Output:**
```
Input A: x_t    [B, 32, 32, 1]   verrauschtes Bild bei Zeitschritt t
Input B: t_emb  [B, 16]          sinusoidales Time-Embedding
Output:  ε̂      [B, 32, 32, 1]   predicted noise
```

---

## Coding Conventions

### TypeScript
- `strict: true`, `noUnusedLocals: true`, `noUnusedParameters: true` — kein Deaktivieren
- Kein `any`. Kein Cast ohne Kommentar warum.
- `const` überall, `let` nur bei Mutation
- Nur named exports, kein default export
- Dateinamen: `camelCase.ts`, Tests: `camelCase.test.ts` neben der Source-Datei

### Canvas & Rendering
- Alle Bilddaten: `Float32Array` in `[-1, 1]` — nie `[0, 255]`
- `renderImageToCanvas()` ist die **einzige** Konvertierungsstelle. Nie duplizieren.
- `setupCanvas()` vor jeder Canvas-Nutzung aufrufen (DPR-Scaling)
- Jede Canvas-Komponente owned ihr DOM-Element. Keine geteilten Canvas-Referenzen.

### TensorFlow.js
- **Alle Tensor-Operationen bleiben auf dem GPU-Backend.** Kein `.data()` / `.array()` im Training-Loop — das verursacht 50× Verlangsamung.
- Tensors mit `tf.tidy()` oder explizitem `.dispose()` verwalten. Memory Leaks crashen den Browser-Tab.
- `tf.randomNormal()` für Rauschen — nicht `Math.random()` in JS-Loop
- Zeit-Input ist immer `[B, 16]` sinusoidales Embedding, nie ein Scalar

### Tests
- Test-Datei neben der Source-Datei: `foo.ts` → `foo.test.ts`
- `describe` + `it` Blöcke — kein nacktes `test()`
- Kein TF.js-Mocking. Nur pure Mathe-Funktionen testen.
- DOM-Tests: happy-dom (konfiguriert in vitest.config.ts). Kein JSDOM.
- Kein `setTimeout` in Tests. Async → await.

---

## Architecture Decision Records

### ADR-001: Sinusoidales Time-Embedding (timeDim=16)
**Was:** Zeit-Input ist ein 16-dim Vektor (8 sin + 8 cos), nicht ein Scalar.
**Warum:** Scalar → Dense(relu) konnte t=0 bis t=400 kaum unterscheiden. HeNormal-Init setzt alle 1024 Neuronen-Breakpoints bei t≈0. Sinusoidal gibt sofort reiches Frequenzspektrum (wie Positional Encoding in Transformers).
**Auswirkung:** Model-Inputs: `[xt(B,32,32,1), tEmb(B,16)]`. Python muss identisch sein. Loss: 0.064→0.038 (-40%) nach diesem Fix.
**Nicht rückgängig machen ohne:** Python-Modell synchron ändern UND public/model/ neu generieren.

### ADR-002: GPU-Only Training Loop
**Was:** Training-Loop hat keinen CPU-Round-Trip.
**Warum:** `.data()` im Loop = 50× Verlangsamung. Bei 2000 Steps inakzeptabel.
**Auswirkung:** `tf.randomNormal()` + `tf.tensor2d()` + alles in `tf.tidy()`. Milestone-Previews außerhalb des Loops.
**Nicht rückgängig machen ohne:** Benchmark-Messung.

### ADR-003: 16/32/64 Filter (war 32/64/128)
**Was:** Encoder 16/32 Filter, Bottleneck 64. Vorher: 32/64/128.
**Warum:** ~500k Params war zu groß für Browser-Fine-Tuning auf 5 Bildern. 118k Params konvergiert schneller.
**Auswirkung:** U-Net Explorer zeigt 16/32/64. Python muss identisch sein.

### ADR-004: 2000 Training Steps (Milestones: 200/800/2000)
**Was:** 2000 Steps (war 500).
**Warum:** 500 Steps = nur ~2 Gradient-Updates pro (t, Bild)-Paar bei 5 Bildern, BatchSize=8, T=400. 2000 Steps = ~8 Updates. Loss stabilisiert sich unter 0.05.
**Auswirkung:** Training ~3-4 Min im Browser. Milestone-Previews bei Step 200/800/2000 (in main.ts und in index.html #ms-200/#ms-800/#ms-final).

### ADR-005: Bottleneck Time-Conditioning
**Was:** Time-Embedding wird additiv direkt in den 8×8×64 Bottleneck injiziert.
**Warum:** Nach 2× MaxPooling war das Time-Signal (im Input projiziert) stark verdünnt. Bottleneck ist wo grobe Entscheidungen fallen — dort braucht das Modell den Zeitschritt.
**Auswirkung:** Zwei Extra-Layer: `Dense(16→64)` + `Reshape(1,1,64)` → `Add(bottleneck)`. Broadcast: [1,1,64] + [B,8,8,64].

### ADR-006: Vanilla TypeScript (kein Framework)
**Was:** Kein React/Vue/Svelte.
**Warum:** Alle UI ist Canvas-basiert (Pixel-Level-Rendering). Ein Component-Framework bringt keinen Nutzen — kein Virtual DOM zu reconcilen, nur Pixel-Arrays.
**Auswirkung:** Komponenten sind plain TS-Klassen. DOM-Verdrahtung in main.ts.

---

## Was Claude NICHT tun soll

- **Kein `.data()` / `.array()` im Training-Loop** — alles in `tf.tidy()`, alles auf dem Backend
- **Keine neuen npm-Abhängigkeiten** ohne starken Grund. TF.js ist schon groß (~8MB).
- **main.ts nicht aufteilen** — absichtlich slim, max. ~150 Zeilen
- **TF.js-Methoden in Tests nicht mocken** — pure Mathe-Funktionen direkt testen
- **Python-Modell nicht ändern** ohne TypeScript-Modell synchron zu ändern — Layer-Namen und Shapes müssen identisch sein für Weight-Loading
- **Kein `setTimeout` in Tests** — async/await verwenden
- **`public/model/*.bin` nicht committen** — gitignored, werden durch convert_to_tfjs.sh regeneriert
- **`noUnusedLocals` / `noUnusedParameters` nicht deaktivieren** — das ist der Lint-Gate
- **Keine Umstrukturierung von src/** ohne dass ein Sprint das explizit fordert

---

## Sprint-Status

| Sprint | Status | Was gebaut wurde |
|--------|--------|-----------------|
| Sprint 1 | ✅ DONE | DatasetPanel, Training, DenoisingStrip, Zoom, Presets, Robotics, FlowMatching |
| Sprint 2 | ✅ DONE | Sinusoidales Embedding, GPU-Loop, 2000 Steps, ForwardDemo, UNetExplorer, Frontend-Redesign |
| Sprint 3 | 🔜 NEXT | Conditional Diffusion |
| Sprint 4 | ⬜ PLANNED | Auto-Steering Demo |
| Sprint 5 | ⬜ PLANNED | DDIM, Deploy |

---

## Nächste Prioritäten (Sprint 3)

In dieser Reihenfolge:

1. **`src/components/conditionalTrainer.ts`** — Conditional U-Net + Class-Embedding + CFG
   - Class-Embedding: `Embedding(numClasses=3, dim=16)` → addiert auf Time-Embedding
   - CFG: 10% null-class dropout beim Training, guidance scale w=2.0 bei Inference
   - Methoden: `trainOnLabeledDataset(images, labels)`, `generateWithClass(classIdx, guidanceScale)`

2. **`src/components/classDatasetPanel.ts`** — 3-Klassen Tabbed UI
   - Tabs: 😊 Happy / 😐 Neutral / 😢 Sad
   - Gleiche Zeichenslots wie DatasetPanel (5 pro Klasse)
   - `getLabeledImages()` → `{images: Float32Array[], labels: number[]}`

3. **Python:** `ml/scripts/train_conditional.py` + `ml/scripts/generate_smiley_dataset.py`
   - Conditional Model → `public/model/conditional-smiley/`

4. **Section 3 in index.html** — "Coming Soon" durch echte UI ersetzen
```

---

### Task 7: Alles committen (Sprint 2 Baseline in main)

**Files:** Alle oben erstellten/geänderten Dateien

- [ ] **Step 1: Tests laufen lassen (sicherstellen dass alles noch grün ist)**

```bash
npm test
```

Expected:
```
Test Files  4 passed (4)
      Tests 33 passed (33)
```

- [ ] **Step 2: Build prüfen**

```bash
npm run build
```

Expected: Kein Fehler, `dist/` wird erstellt.

- [ ] **Step 3: TypeCheck**

```bash
npx tsc --noEmit
```

Expected: Keine Ausgabe (=kein Fehler).

- [ ] **Step 4: Staged alles relevante**

```bash
# Alle Modifikationen + Deletions (tracked files) stagen
git add -u

# Neue untracked Dateien explizit stagen
git add CLAUDE.md README.md docs/ public/model/ .gitignore src/ vitest.config.ts index.html package.json package-lock.json ml/

# Status prüfen — model/ sollte NICHT erscheinen (gitignored)
git status
```

Prüfe die Ausgabe:
- `docs/REQUIREMENTS.md`, `docs/ARCHITECTURE.md`, `docs/TEST_PLAN.md`, `docs/superpowers/` sollen grün sein (new file)
- `CLAUDE.md`, `README.md` sollen grün sein (new file)
- `REQUIREMENTS.md`, `ARCHITECTURE.md` (root) sollen als "deleted" grün sein
- `ONNX`-Dateien sollen als "deleted" grün sein (aus Step 1: git rm)
- `model/` soll NICHT erscheinen (gitignored)
- `public/model/model.json` soll grün sein (new file — wird vom Browser für pretrained weights gebraucht)

- [ ] **Step 5: Committen**

```bash
git commit -m "$(cat <<'EOF'
feat: Sprint 2 complete — docs reorganisation, CI/CD setup, frontend redesign

- Delete ONNX experiments and unused Python scripts from root
- Move REQUIREMENTS.md + ARCHITECTURE.md to docs/ (updated to reflect Sprint 2)
- Add docs/TEST_PLAN.md: all 33 tests documented with purpose and rationale
- Add CLAUDE.md: session prep, coding conventions, ADRs, what NOT to do
- Add README.md: GitHub-facing quick start and project overview
- Sinusoidal time embedding (timeDim=16), GPU-only training loop
- 2000 training steps (milestones 200/800/2000)
- Dark Amber Phosphor frontend redesign (Syne + DM Sans fonts)
- GitHub Actions CI/CD (see Chunk 2)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 6: Push zu main**

```bash
git push origin main
```

---

## Chunk 2: GitHub Actions CI/CD

### Task 8: CI Workflow erstellen

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: .github/workflows Verzeichnis anlegen**

```bash
mkdir -p .github/workflows
```

- [ ] **Step 2: ci.yml erstellen**

```yaml
name: CI

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  ci:
    name: TypeCheck + Test + Build
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: 22
          cache: npm

      - name: Install dependencies
        run: npm ci

      - name: TypeCheck (lint)
        run: npx tsc --noEmit

      - name: Tests
        run: npm test

      - name: Build
        run: npm run build
```

- [ ] **Step 3: Committen**

```bash
git add .github/
git commit -m "$(cat <<'EOF'
ci: add GitHub Actions CI pipeline (typecheck + test + build)

Runs on every PR to main and on push to main.
Steps: tsc --noEmit (lint) → vitest run → vite build

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4: Push**

```bash
git push origin main
```

- [ ] **Step 5: Verify CI läuft auf GitHub**

Öffne: https://github.com/zornalex/diffusion_explorer/actions

Expected: Ein grüner Workflow-Run für den letzten Push auf main.

---

### Task 9: Branch Protection (manuell im GitHub UI)

CI-Pipeline läuft, jetzt Branch Protection aktivieren.

**Dies muss manuell vom User im GitHub UI gemacht werden — automatisieren via gh CLI ist möglich aber erfordert Admin-Token.**

- [ ] **Step 1: User-Anleitung**

Gehe zu: https://github.com/zornalex/diffusion_explorer/settings/branches

Klicke **"Add branch protection rule"**

Trage ein:
- Branch name pattern: `main`

Aktiviere folgende Checkboxen:
- ✅ **Require a pull request before merging**
  - ✅ Require approvals: 0 (kein Reviewer nötig — solo project)
- ✅ **Require status checks to pass before merging**
  - Suche und wähle: `TypeCheck + Test + Build` (der Job-Name aus ci.yml)
  - ✅ Require branches to be up to date before merging
- ✅ **Do not allow bypassing the above settings**

Klicke **"Create"**

- [ ] **Step 2: Verify — Feature-Branch erstellen und PR öffnen**

```bash
git checkout -b feature/test-branch-protection
echo "# test" >> docs/TEST_PLAN.md
git add docs/TEST_PLAN.md
git commit -m "test: verify branch protection works"
git push origin feature/test-branch-protection
```

Öffne einen PR auf GitHub: https://github.com/zornalex/diffusion_explorer/pulls

Expected:
- CI startet automatisch auf dem PR
- Merge-Button ist deaktiviert bis CI grün ist
- Nach CI grün: Merge-Button aktiviert

- [ ] **Step 3: Test-Branch aufräumen (nach Verify)**

```bash
git checkout main
git branch -d feature/test-branch-protection
git push origin --delete feature/test-branch-protection
```

---

## Fertig — Was jetzt gilt

Nach Abschluss dieses Plans:

```
main (protected)
  → nur via PR mergbar
  → CI muss grün sein: tsc --noEmit + vitest run + vite build

feature/sprint3-conditional   ← nächster Branch für Sprint 3
feature/docs-xyz              ← für Docs-Änderungen
```

Jeder neue Branch:
```bash
git checkout main
git pull origin main
git checkout -b feature/<name>
# ... Arbeit ...
git push origin feature/<name>
# → PR auf GitHub → CI läuft → Merge
```
