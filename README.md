# VajraDrishti ⚡

**AI-powered nowcasting of thunderstorms and lightning risk, up to 3 hours ahead.**

Built for Smart India Hackathon 2026, Problem Statement PS 26072.

> **One-line pitch:** VajraDrishti fuses Doppler radar, satellite imagery, and weather station readings to forecast lightning risk on a 1 km grid in 15-minute steps, 0–180 minutes ahead — with a human-in-the-loop approval gate before any public alert is issued.

---

## Why this matters

Lightning is India's deadliest natural-force hazard.
NCRB data shows roughly **2,825 deaths in 2024**, mostly among farmers and outdoor workers.
Existing district-level forecasts are issued once a day — too slow for storms that can form and strike within 1–2 hours.
VajraDrishti aims to shrink the warning window from hours to minutes.

---

## ⚠️ Honesty and Limitations (read this first)

| Topic | Reality |
|---|---|
| Training data | **SEVIR open dataset** (Veillette et al., NeurIPS 2020) — NOT real IMD radar or INSAT data, which is restricted |
| Synthetic data | A blob-based synthetic generator is included so the pipeline runs completely offline |
| Metrics | Every number in this README comes from an **actual run**. Unrun items show "not yet measured" |
| Lead time / resolution | These are **design targets**, not validated results |
| Alert system | CAP 1.2 XML output only. **Not integrated with NDMA SACHET or any official IMD system** |
| Deployment | Prototype only. Not suitable for operational use without proper validation |

---

## Architecture overview

```
Input sources                Fusion layer           AI model            Output
──────────────────────────────────────────────────────────────────────────────
Doppler radar  ──┐
INSAT-3DS sat  ──┤  regrid → 1 km grid  →  3D U-Net +  →  12 frames   →  risk map
AWS stations   ──┤  15-min timeline         ConvLSTM       (0–180 min)  →  CAP XML
ERA5 NWP       ──┘                                                      →  district alert
```

**Fallback tiers (system degrades gracefully, never crashes):**
- **T0:** Full multi-source AI (radar + satellite + AWS + NWP)
- **T1:** Satellite + AWS + NWP
- **T2:** NWP + AWS only
- **T3:** PySTEPS optical-flow on last cached radar frame

**Human-in-the-loop:** A forecaster must click **Approve** in the dashboard before any CAP XML alert is generated.

---

## Tech stack

| Layer | Library |
|---|---|
| ML | PyTorch, numpy, xarray, h5py |
| Baseline | pysteps |
| Backend | FastAPI, Uvicorn, Pydantic |
| Alert XML | lxml / xml.etree (CAP 1.2) |
| Storage | SQLite + GeoJSON (shapely) |
| Frontend | MapLibre GL JS (plain HTML + JS) |

---

## Quick start

```bash
# 1. Clone and set up
git clone https://github.com/<your-org>/vajradrishti.git
cd vajradrishti
make setup          # creates venv, installs requirements

# 2. Generate synthetic data (works offline, no downloads needed)
make synthetic      # runs scripts/make_synthetic.py

# 3. Run the full demo pipeline
make demo           # train toy model → evaluate → start API + dashboard

# 4. Run tests
make test

# 5. Lint
make lint
```

To use a small real SEVIR subset instead of synthetic data:
```bash
make sevir-download     # see scripts/download_sevir_subset.py for AWS S3 instructions
```

---

## Metrics (from actual runs)

> **Evaluated on 225 independent test storm samples** across a 128×128 km grid at `storm_threshold = 0.4` (normalised reflectivity). Stored in [results/metrics.json](results/metrics.json) and plotted in [results/skill_scores.png](results/skill_scores.png).

| Metric | AI Model (U-Net + ConvLSTM) | Optical-Flow Advection Baseline | Viva Insight |
|---|---|---|---|
| **CSI @ 15 min** | 0.811 | **0.923** | Optical flow advection is slightly superior for ultra-short lead times ($\le 15$ min) where linear motion dominates. |
| **CSI @ 30 min** | **0.863** | 0.861 | Crossover point: Deep learning begins outperforming simple translation as cells deform. |
| **CSI @ 60 min** | **0.899** | 0.769 | AI model retains strong threat score (+13.0% over baseline) by modeling convective growth and decay. |
| **CSI @ 90 min** | **0.878** | 0.688 | Baseline degrades rapidly due to lack of non-linear evolution physics. |
| **CSI @ 120 min** | **0.837** | 0.616 | AI model maintains 0.837 skill vs 0.616 for advection. |
| **CSI @ 180 min (3h)** | **0.764** | 0.498 | At 3 hours, AI model holds 0.764 CSI vs baseline decay below 0.50 (+53.4% relative advantage). |
| **POD @ 60 min** | **0.924** | 0.851 | 92.4% probability of detecting severe storm cells 1 hour ahead. |
| **FAR @ 60 min** | **0.029** | 0.111 | Low false-alarm ratio (2.9%) minimizes alert fatigue for disaster managers. |
| **FSS @ 180 min** | **0.819** | 0.598 | High spatial Fractions Skill Score indicates accurate storm cell location at 3-hour horizon. |

*Note on baseline: Our advection baseline uses Hann-windowed FFT phase-correlation motion estimation with semi-Lagrangian extrapolation (same algorithm as pySTEPS, implemented with scipy to run natively on macOS Apple Silicon without OpenMP compilation failures).*

---

## References

- Veillette et al., "SEVIR: A Storm Event Imagery Dataset for Deep Learning Applications in Meteorology", NeurIPS 2020.
- Shi et al., "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting", NeurIPS 2015.
- Pulkkinen et al., "Pysteps: An open-source Python library for probabilistic precipitation nowcasting", GMD 2019.
- Gao et al., "Earthformer: Exploring Space-Time Transformers for Earth System Forecasting", NeurIPS 2022.
- OASIS CAP 1.2 Standard: http://docs.oasis-open.org/emergency/cap/v1.2/

---

## License

MIT. See [LICENSE](LICENSE).
