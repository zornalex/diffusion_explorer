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
