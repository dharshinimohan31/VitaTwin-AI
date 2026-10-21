# VitaTwin AI — Safety & Scope

**VitaTwin AI is a research and educational prototype built for the Happiest Health
Digital Twin Challenge 2026. It is not a medical device, not a diagnostic tool, and not
a substitute for professional medical care.**

## What this means concretely

- Every prediction is an **experimental model output**, computed by a baseline Random
  Forest trained on synthetic data (see `docs/dataset.md`, `docs/model.md`). It is not
  a clinical measurement or a diagnosis.
- Risk categories (LOW/MODERATE/HIGH) use **documented, fixed probability thresholds**
  (`src/config/settings.py`) chosen to make all three categories reachable on this
  dataset — they are not derived from, and should not be read as, a validated clinical
  risk-scoring system.
- The What-If Simulator shows how the **trained model's own output** changes under a
  hypothetical input. It does **not** claim that performing the simulated action will
  produce that result in a real patient. Every simulation response carries this
  disclaimer verbatim (`src/simulation/whatif.py::DISCLAIMER`).
- Preventive recommendations are **decision-support suggestions** ("consider...",
  "worth noting..."), never instructions, and a HIGH predicted risk always includes a
  prompt to involve a healthcare professional rather than a specific clinical action.
- **No real patient data is used anywhere in this project.** All data is either
  synthetically generated (`data_source=synthetic`, see `docs/dataset.md`) or, in
  future work, would be a disclosed real-data substitution — never silently presented
  as real.
- This prototype has **not been clinically validated** and makes no claim of clinical
  accuracy, regulatory approval, or fitness for real-world clinical use.

## Displayed to the user

The research/prototype disclaimer above is shown at the top of the Streamlit dashboard
on every screen that displays a prediction, risk category, or simulation result
(`src/dashboard/app.py`), and is returned in the `/health` endpoint of the API
(`src/api/main.py`) so any programmatic consumer sees it too.
