# Prosody in audio-language models: CKA analysis

Implementation following [plan.md](plan.md): datasets (Expresso, ESD) and CKA-based prosody-contrastive analysis for conversational and encoder→projector→LLM models.

## Data

- **Expresso** (you have it): read speech, 7 styles × 4 speakers, parallel same-text + contrastive emphasis. Paths set via `EXPRESSO_ROOT` (default: parent `../expresso`).
- **ESD** (Emotional Speech Dataset): 5 emotions × 10 speakers. Path via `ESD_ROOT`.

## Layout

- `data/` – data loaders (Expresso, ESD)
- `cka/` – Centered Kernel Alignment utilities
- `scripts/` – run extraction and CKA
- `config/` – paths and model config (optional)

## Verify in a notebook

Open **`verify_pipeline.ipynb`** in Jupyter. It runs the pipeline step-by-step (Expresso/ESD loaders, parallel groups, CKA on dummy features). Run cells in order; the first cell adds the `colm` folder to `PYTHONPATH` based on the current working directory.

## Quick start (command line)

Run from the **project root** (parent of `colm/`, e.g. `bkoduru/`):

```bash
# Create env and install
python -m venv .venv && source .venv/bin/activate  # or conda
pip install -r requirements.txt

# From project root, set PYTHONPATH to the colm project folder
cd /path/to/bkoduru
export PYTHONPATH=/path/to/bkoduru/colm:$PYTHONPATH

# List Expresso parallel same-text groups (uses EXPRESSO_ROOT or --root)
export EXPRESSO_ROOT=/ocean/projects/cis220031p/bkoduru/expresso
python -m colm.scripts.list_expresso_pairs --max-groups 5

# Run CKA (placeholder extractor; plug in your model later)
python -m colm.scripts.run_cka --dataset expresso --max-groups 20 --output cka_expresso.npz
python -m colm.scripts.run_cka --dataset esd --max-groups 20 --output cka_esd.npz

# Or use pre-built pair manifests (e.g. interspeech/expresso_local_2.json, esp/esp_local.json)
python -m colm.scripts.run_cka --manifest /path/to/expresso_local_2.json --max-pairs 500 --output cka_expresso.npz
python -m colm.scripts.run_cka --manifest /path/to/esp_local.json --max-pairs 500 --output cka_esd.npz
```

## Plan summary

- **Tier 1**: Moshi (Mimi codebooks, Temporal Transformer, Inner Monologue)
- **Tier 2**: LLaMA-Omni, Freeze-Omni (Whisper→adapter→LLM)
- **Tier 3**: pGSLM, SpiRit-LM Expressive (explicit prosody streams)

Evaluation: ESD for emotion-contrastive; Expresso for prosodic style and emphasis.
