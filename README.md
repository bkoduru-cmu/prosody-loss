# Prosody-Loss: CKA Analysis for Audio–Language Models

**Prosody-contrastive analysis** for conversational and encoder→projector→LLM models: same-text, different-style/emotion pairs from **Expresso** and **ESD**, with **Centered Kernel Alignment (CKA)** to measure where prosodic information is preserved or lost across layers.

Implementation follows the strategy in [plan.md](plan.md): datasets, model tiers (Moshi, LLaMA-Omni, Freeze-Omni, pGSLM, SpiRit-LM, etc.), and CKA-based evaluation.

---

## Data

| Dataset | Description | Environment / default path |
|--------|-------------|----------------------------|
| **Expresso** | Read speech, 7 styles × 4 speakers; parallel same-text + contrastive emphasis | `EXPRESSO_ROOT` (default: `../expresso`) |
| **ESD** | Emotional Speech Dataset: 5 emotions × 10 speakers | `ESD_ROOT` (default: `../Emotion Speech Dataset`) |

---

## Installation

```bash
# Clone and enter repo
git clone https://github.com/bkoduru-cmu/prosody-loss.git
cd prosody-loss

# Create environment
python -m venv .venv && source .venv/bin/activate   # or: conda create -n prosody-loss python=3.10 && conda activate prosody-loss
pip install -r requirements.txt

# So the colm package is importable (from repo root)
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

Optional: set dataset roots before running scripts:

```bash
export EXPRESSO_ROOT=/path/to/expresso
export ESD_ROOT=/path/to/Emotion\ Speech\ Dataset
```

---

## Project layout

```
prosody-loss/
├── README.md           # This file
├── RUNBOOK.md          # Full pipeline runbook (manifests, models, CKA)
├── plan.md             # Datasets, model tiers, CKA strategy
├── requirements.txt
├── colm/               # Python package
│   ├── config.py       # Paths (EXPRESSO_ROOT, ESD_ROOT) and constants
│   ├── data/           # Expresso & ESD loaders, pair manifests
│   ├── cka/            # linear_cka, kernel_cka
│   ├── extractors/     # Model extractors (e.g. Qwen2-Audio)
│   └── scripts/        # CLI entrypoints
├── verify_pipeline.ipynb   # Step-by-step pipeline check (data + CKA)
├── expresso_umap.ipynb    # UMAP visualization by style/speaker
└── expresso_umap_by_style_and_speaker.png
```

---

## How to run each file

All commands below are meant to be run from the **repo root** (`prosody-loss/`), with `PYTHONPATH` set as above.

---

### 1. `colm/scripts/list_expresso_pairs.py`

**Purpose:** List Expresso parallel same-text groups (and optionally emphasis-only). Verifies data paths.

**Run:**

```bash
# Default root from EXPRESSO_ROOT or config
python -m colm.scripts.list_expresso_pairs --max-groups 5

# Custom root
python -m colm.scripts.list_expresso_pairs --root /path/to/expresso --max-groups 10

# Only emphasis subset
python -m colm.scripts.list_expresso_pairs --emphasis-only --max-groups 5

# Require at least 3 styles per group
python -m colm.scripts.list_expresso_pairs --min-styles 3 --max-groups 5
```

**Options:** `--root`, `--max-groups`, `--emphasis-only`, `--min-styles`.

---

### 2. `colm/scripts/build_expresso_all_manifest.py`

**Purpose:** Build a JSON manifest of all Expresso read-speech files: `path`, `style`, `speaker` for every WAV under the read-speech tree.

**Run:**

```bash
# Default: writes expresso_all.json in current directory
python -m colm.scripts.build_expresso_all_manifest

# Custom root and output path
python -m colm.scripts.build_expresso_all_manifest --root /path/to/expresso --out ./manifests/expresso_all.json
```

**Options:** `--root`, `--out`. Output is a JSON list of `{"path", "style", "speaker"}`.

---

### 3. `colm/scripts/run_cka.py`

**Purpose:** Run CKA on prosody-contrastive pairs: load pairs (from dataset roots or a manifest), extract features (placeholder by default; plug in your model), compute linear CKA between same-text different-style/emotion representations, save results to `.npz`.

**Run:**

**From dataset roots (Expresso or ESD):**

```bash
# Expresso (uses EXPRESSO_ROOT or --root)
python -m colm.scripts.run_cka --dataset expresso --max-groups 20 --output cka_expresso.npz

# ESD (uses ESD_ROOT or --root)
python -m colm.scripts.run_cka --dataset esd --max-groups 20 --output cka_esd.npz

# Custom root and limits
python -m colm.scripts.run_cka --dataset expresso --root /path/to/expresso --max-groups 50 --min-styles 2 --output cka_expresso.npz
```

**From a pre-built pair manifest (e.g. from another repo):**

```bash
python -m colm.scripts.run_cka --manifest /path/to/expresso_local_2.json --max-pairs 500 --output cka_expresso.npz
python -m colm.scripts.run_cka --manifest /path/to/esp_local.json --max-pairs 500 --output cka_esd.npz
```

**Options:** `--dataset` (expresso | esd), `--manifest`, `--root`, `--max-groups`, `--max-pairs`, `--min-styles` / `--min-emotions`, `--feat-dim`, `--output`.  
**Note:** The script uses a dummy feature extractor by default; replace `dummy_extract` in `run_cka.py` with your model’s forward pass (e.g. encoder/projector hidden states) to get real CKA scores.

---

### 4. `verify_pipeline.ipynb`

**Purpose:** Verify the full pipeline step-by-step: Expresso/ESD loaders, parallel groups, and CKA on dummy features. Optional: load pair manifests.

**How to run:**

1. Open in Jupyter (or VS Code): from repo root, `jupyter notebook verify_pipeline.ipynb` or open the file in the editor.
2. Set kernel to the environment where you installed `requirements.txt`.
3. Run cells in order. The first cell adds the repo root to `sys.path` so `import colm` works (works whether your cwd is repo root or `colm/`).

No command-line invocation; run interactively.

---

### 5. `expresso_umap.ipynb`

**Purpose:** Load Expresso utterances (from `expresso_all.json` or by scanning the dataset), extract a fixed-size feature per utterance (e.g. mel-spectrogram stats), run UMAP to 2D, and plot points colored by **style** and optionally **speaker** to see if prosodic styles cluster.

**How to run:**

1. Install extra deps in your environment (run once in the notebook or in the terminal):
   ```bash
   pip install librosa umap-learn matplotlib tqdm
   ```
2. Open `expresso_umap.ipynb` in Jupyter (or VS Code).
3. If using the full manifest: ensure `expresso_all.json` exists (e.g. from `build_expresso_all_manifest.py`) and set the path in the notebook (e.g. `interspeech/expresso_all.json` or your chosen path).
4. Run all cells in order.

No CLI; run interactively.

---

## Quick reference

| What you want | Command or file |
|---------------|------------------|
| List Expresso same-text groups | `python -m colm.scripts.list_expresso_pairs ...` |
| Build full Expresso path manifest | `python -m colm.scripts.build_expresso_all_manifest ...` |
| Run CKA (Expresso/ESD or manifest) | `python -m colm.scripts.run_cka ...` |
| Verify data loaders + CKA | Run `verify_pipeline.ipynb` |
| UMAP by style/speaker | Run `expresso_umap.ipynb` (after `pip install librosa umap-learn matplotlib tqdm`) |

---

## Plan summary

- **Tier 1:** Moshi (Mimi codebooks, Temporal Transformer, Inner Monologue).
- **Tier 2:** LLaMA-Omni, Freeze-Omni (Whisper→adapter→LLM).
- **Tier 3:** pGSLM, SpiRit-LM Expressive (explicit prosody streams).

**Evaluation:** ESD for emotion-contrastive analysis; Expresso for prosodic style and emphasis. See [plan.md](plan.md) and [RUNBOOK.md](RUNBOOK.md) for full details and model-by-model run instructions.
