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
✓ src/core/config.test.ts           (7 tests)
✓ src/components/renderUtils.test.ts (5 tests)
✓ src/core/diffusion.test.ts        (11 tests)
✓ src/data/presets.test.ts          (10 tests)

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
| 5 | `different smiley variants are not identical (happy vs sad)` | `sum(\|happy - sad\|) > 0` | Stellt sicher dass die Presets tatsächlich unterschiedliche Bilder sind. Gleiche Bilder → Modell kann keine Distribution lernen. |
| 6 | `SMILEY_LABELS has same length as SMILEY_PRESETS` | `SMILEY_LABELS.length === 8` | Labels werden parallel zu Presets verwendet (UI-Anzeige). Unterschiedliche Längen → Array-Out-of-Bounds. |
| 7 | `CLASS_PRESETS has 3 classes (happy, neutral, sad)` | `CLASS_PRESETS.length === 3` | Conditional Diffusion (Sprint 3) erwartet genau 3 Klassen (Index 0=happy, 1=neutral, 2=sad). |
| 8 | `each class has at least one image` | `CLASS_PRESETS[i].length > 0` für alle i | Leere Klasse → Training kann keine Samples aus dieser Klasse ziehen → Gradient-Fehler. |
| 9 | `each class image is Float32Array of correct size` | Float32Array + length===1024 für alle Klassenbilder | Gleiche Constraints wie bei SMILEY_PRESETS. Konsistenz zwischen unconditional und conditional Dataset. |
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
