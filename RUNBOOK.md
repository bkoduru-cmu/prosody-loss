# Runbook: Running the full prosody CKA plan for all models

This maps **every step in [plan.md](plan.md)** to where and how to run it. Use one place for data + CKA, and one place per model (or one orchestration notebook) for extraction.

---

## Pipeline steps (same for every model)

| Step | What | Where |
|------|------|--------|
| **1. Data** | Load same-text, different-style/emotion pairs | **Manifests**: `interspeech/expresso_local_2.json` (Expresso), `interspeech/esp/esp_local.json` (ESD). Or colm: `build_expresso_parallel_groups` / `build_esd_parallel_groups` / `load_*_pairs_manifest`. |
| **2. Extract** | For each (audio1, audio2), run model and get representations (encoder, projector, LLM, or model-specific) | **Per-model** (see table below). |
| **3. CKA** | Compute CKA (or correlation/cosine) between same-pair representations | **colm**: `linear_cka`, `kernel_cka` in `colm.cka`. |
| **4. Aggregate** | Per style-pair or overall mean ± std, save and compare across models | Your notebook or `run_prosody_plan.ipynb`. |

---

## Where each step runs today

| Step | File / location |
|------|-----------------|
| Load Expresso pairs | `expresso_local_2.json` or `colm.data.pair_manifests.load_expresso_pairs_manifest` |
| Load ESD pairs | `esp/esp_local.json` or `colm.data.pair_manifests.load_esd_pairs_manifest` |
| CKA | `colm.cka.linear_cka` (used in `colm/verify_pipeline.ipynb`, `colm/scripts/run_cka.py`) |
| **Qwen2-Audio** (encoder / projector / LLM) | **`interspeech/esp/esp.ipynb`** — loads ESD manifest, extracts encoder attention, projector output, LLM attention; computes correlation/cosine; saves `style_pair_results_*.json`. |
| **Qwen2-Audio on Expresso** | **`interspeech/icml.ipynb`** — intended for Expresso (dataset download there; can switch to local manifest + same extractor as esp). |
| Other models | Not implemented yet; add extractors in **`interspeech/run_prosody_plan.ipynb`** (or separate notebooks). |

---

## Models from the plan (what to run where)

| Tier | Model | Same pipeline? | Where to run / status |
|------|--------|-----------------|------------------------|
| Baseline | **Qwen2-Audio-7B** | Yes (encoder → projector → LLM) | **ESD**: run **`esp/esp.ipynb`** (uses `esp_local.json`). **Expresso**: use `expresso_local_2.json` with the same extractor logic from esp.ipynb (e.g. in `run_prosody_plan.ipynb`). |
| Tier 2 | **LLaMA-Omni** | Yes (Whisper → adapter → LLM) | Add extractor in `run_prosody_plan.ipynb`; same manifests + CKA. |
| Tier 2 | **Freeze-Omni** | Yes (frozen LLM) | Add extractor; same manifests + CKA. |
| Tier 1 | **Moshi** | No (Mimi codebooks + Temporal Transformer) | Add extractor (codebook levels + hidden states); same manifests + CKA. |
| Tier 3 | **pGSLM** | No (F0 + duration streams) | Add extractor; same manifests + CKA. |
| Tier 3 | **SpiRit-LM Expressive** | No (pitch + style tokens) | Add extractor; same manifests + CKA. |

---

## How to “run every single step for all models”

1. **One orchestration notebook**: **`interspeech/run_prosody_plan.ipynb`**
   - Sets `PYTHONPATH` / imports so **colm** is available (manifests + CKA).
   - Defines paths: `expresso_local_2.json`, `esp/esp_local.json`.
   - For **each model** in a list:
     - Load that model (or skip if not implemented).
     - Load pairs from the chosen manifest (Expresso and/or ESD).
     - For each pair: extract representations → compute CKA (or reuse correlation/cosine) → store.
     - Aggregate and save (e.g. `results_<model>_<dataset>.json`).
   - **Qwen2-Audio**: reuse the same extraction logic as in **esp.ipynb** (encoder, projector, LLM); optionally add CKA in addition to correlation.
   - **Other models**: add a cell or function “extract_llama_omni”, “extract_moshi”, etc.; same loop, same CKA step.

2. **Run order**
   - **Data**: Manifests are already built; no separate “run” step.
   - **Qwen2-Audio on ESD**: Run **`esp/esp.ipynb`** (already does full pipeline for ESD).
   - **Qwen2-Audio on Expresso** + **all models (as you add them)** on both datasets: Run **`run_prosody_plan.ipynb`** (add one block per model; use colm for manifests + CKA).

3. **Optional: CKA in esp.ipynb**
   - esp.ipynb currently uses **correlation** (encoder, LLM) and **cosine similarity** (projector). To align with the plan’s “CKA” wording, you can additionally compute **linear CKA** on the same representations (e.g. `last_hidden_state` or attention maps) in the same notebook or in `run_prosody_plan.ipynb` and report both.

---

## Quick reference: file roles

| File | Role |
|------|------|
| **colm/plan.md** | Plan: datasets, models, tiers, CKA strategy. |
| **colm/verify_pipeline.ipynb** | Verify colm data loaders + CKA (dummy features); optional manifest section. |
| **colm/scripts/run_cka.py** | CLI: CKA from dataset roots or from `--manifest` (dummy extractor; plug in real model later). |
| **colm/data/pair_manifests.py** | Load `expresso_local_2.json` / `esp_local.json` for pair lists. |
| **interspeech/esp/esp.ipynb** | **Qwen2-Audio on ESD**: load model, esp_local.json, extract encoder/projector/LLM, correlation/cosine, save results. |
| **interspeech/icml.ipynb** | Expresso + Qwen2-Audio (dataset download; can be switched to local manifest + same extractor). |
| **interspeech/run_prosody_plan.ipynb** | **Single place to run the full pipeline for all models**: manifests, loop over models, extract + CKA, aggregate. **Section 5** lists **all models from the plan** (Moshi, LLaMA-Omni, Freeze-Omni, Mini-Omni2, pGSLM, SpiRit-LM Exp, dGSLM, GLM-4-Voice, VITA-Audio, Ichigo) with repo links and what to extract; each has a stub cell to implement the extractor and call `run_cka_for_model`. |

Running “every single step for all models” means: run **esp.ipynb** for Qwen2 + ESD, and **run_prosody_plan.ipynb** for all models (and both Expresso and ESD) as you add each extractor.
