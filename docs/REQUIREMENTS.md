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
