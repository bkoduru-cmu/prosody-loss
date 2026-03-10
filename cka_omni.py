"""
CKA Analysis + State Extraction - Qwen2.5-Omni-7B
Full paired dataset, resumable, saves incrementally after every pair.
Also saves mean-pooled hidden states per clip for linear probing (no separate extraction pass needed).

Usage: just run it. It picks up where it left off automatically.
Set SAMPLES_PER_RUN = None to run all remaining pairs.
Plot CKA  : python plot_cka.py --results cka_omni_output/global_results.json
Train probe: python probe.py --states cka_omni_output/hidden_states.jsonl
"""

import os
import torch
import numpy as np
import json
import gc
from tqdm import tqdm
from collections import defaultdict
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
MODEL_ID        = "Qwen/Qwen2.5-Omni-7B"
PAIRS_FILE      = "/ocean/projects/cis220031p/bkoduru/colm/manifests/expresso_full_pairs.json"
CACHE_DIR       = "/ocean/projects/cis220031p/bkoduru/huggingface/hub"
OUTPUT_DIR      = "cka_omni_output"
SAMPLES_PER_RUN = None        # None = all remaining, or set e.g. 500
EXCLUDED_STYLES = {"singing", "narration"}

os.makedirs(OUTPUT_DIR, exist_ok=True)

PROCESSED_IDS_FILE  = f"{OUTPUT_DIR}/processed_ids.json"
GLOBAL_RESULTS_FILE = f"{OUTPUT_DIR}/global_results.json"
OUTPUTS_FILE        = f"{OUTPUT_DIR}/model_outputs.json"
STATES_FILE         = f"{OUTPUT_DIR}/hidden_states.jsonl"   # for probe.py

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
print(f"Loading {MODEL_ID}...")
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    cache_dir=CACHE_DIR,
    local_files_only=True,
)
processor = Qwen2_5OmniProcessor.from_pretrained(
    MODEL_ID,
    cache_dir=CACHE_DIR,
    local_files_only=True,
)
model.eval()
print("Model loaded.")

# ─────────────────────────────────────────────
# CKA
# ─────────────────────────────────────────────
def centered_kernel_alignment(X, Y):
    if X.ndim == 1: X = X.reshape(-1, 1)
    if Y.ndim == 1: Y = Y.reshape(-1, 1)
    if X.shape[0] != Y.shape[0]:
        m = min(X.shape[0], Y.shape[0])
        X, Y = X[:m], Y[:m]
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    XXT = X @ X.T
    YYT = Y @ Y.T
    hsic = np.trace(XXT @ YYT)
    norm = np.sqrt(np.trace(XXT @ XXT) * np.trace(YYT @ YYT))
    return float(hsic / (norm + 1e-8))

# ─────────────────────────────────────────────
# FEATURE EXTRACTION + GENERATION
# ─────────────────────────────────────────────
def get_features_and_output(audio_path):
    from qwen_omni_utils import process_mm_info

    conversation = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user",   "content": [
            {"type": "audio", "audio": audio_path},
            {"type": "text",  "text": "What is the person saying? Describe the tone too."},
        ]},
    ]

    text = processor.tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    inputs = processor(
        text=text, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True,
    ).to(model.device)

    input_ids       = inputs["input_ids"][0]
    tokens          = processor.tokenizer.convert_ids_to_tokens(input_ids.tolist())
    audio_positions = [i for i, t in enumerate(tokens) if t and "<|AUDIO|>" in t]

    enc_states, dec_states, proj_state = [], [], []
    hooks = []

    for layer in model.thinker.audio_tower.layers:
        def enc_hook(m, inp, out, s=enc_states):
            h = out[0] if isinstance(out, tuple) else out
            s.append(h.squeeze(0).detach().cpu().float().numpy())
        hooks.append(layer.register_forward_hook(enc_hook))

    hooks.append(model.thinker.audio_tower.proj.register_forward_hook(
        lambda m, inp, out: proj_state.append(
            out.squeeze(0).detach().cpu().float().numpy()
        )
    ))

    seq_len = input_ids.shape[0]
    for layer in model.thinker.model.layers:
        def dec_hook(m, inp, out, s=dec_states, pos=audio_positions, sl=seq_len):
            h = out[0] if isinstance(out, tuple) else out
            h = h.squeeze(0)
            if h.shape[0] == sl and pos:
                s.append(h[pos].detach().cpu().float().numpy())
        hooks.append(layer.register_forward_hook(dec_hook))

    with torch.no_grad():
        model.thinker(**inputs, output_hidden_states=True, return_dict=True)

    for h in hooks:
        h.remove()

    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=100, return_audio=False)
    text_out = processor.tokenizer.decode(
        out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )

    return {
        "enc":    enc_states,                          # 32 × (T, 1280) — full states for CKA
        "prj":    proj_state[0] if proj_state else None,
        "dec":    dec_states,                          # 28 × (n_audio_tokens, 3584)
        "output": text_out,
    }

# ─────────────────────────────────────────────
# INCREMENTAL SAVE (CKA + outputs)
# ─────────────────────────────────────────────
def save_all(processed_ids, global_results, all_outputs):
    with open(PROCESSED_IDS_FILE, "w") as f:
        json.dump(list(processed_ids), f)

    global_agg = {}
    for pair, layers in global_results.items():
        global_agg[pair] = {}
        for layer, vals in layers.items():
            if vals:
                global_agg[pair][layer] = {
                    "mean": float(np.mean(vals)),
                    "std":  float(np.std(vals)),
                    "n":    len(vals),
                }
    with open(GLOBAL_RESULTS_FILE, "w") as f:
        json.dump({
            "total_processed":  len(processed_ids),
            "total_pairs":      len(all_pairs),
            "aggregated_stats": global_agg,
            "raw_results": {p: dict(l) for p, l in global_results.items()},
        }, f)

    with open(OUTPUTS_FILE, "w") as f:
        json.dump(all_outputs, f, indent=2)

# ─────────────────────────────────────────────
# LOAD PAIRS
# ─────────────────────────────────────────────
print("\nLoading pairs...")
with open(PAIRS_FILE) as f:
    all_pairs = json.load(f)

all_pairs = [
    p for p in all_pairs
    if p["style1"] not in EXCLUDED_STYLES and p["style2"] not in EXCLUDED_STYLES
]
print(f"Pairs after filtering: {len(all_pairs)}")

# ─────────────────────────────────────────────
# RESUME STATE
# ─────────────────────────────────────────────
processed_ids = set()
if os.path.exists(PROCESSED_IDS_FILE):
    with open(PROCESSED_IDS_FILE) as f:
        processed_ids = set(json.load(f))
    print(f"Already processed: {len(processed_ids)} pairs")
else:
    print("Starting fresh.")

global_results = defaultdict(lambda: defaultdict(list))
if os.path.exists(GLOBAL_RESULTS_FILE):
    with open(GLOBAL_RESULTS_FILE) as f:
        saved = json.load(f)
    for pair, layers in saved.get("raw_results", {}).items():
        for layer, vals in layers.items():
            global_results[pair][layer] = vals
    print("Loaded existing global results.")

all_outputs = []
if os.path.exists(OUTPUTS_FILE):
    with open(OUTPUTS_FILE) as f:
        all_outputs = json.load(f)

# Load already-seen clip IDs from states file (deduplication)
seen_clips = set()
if os.path.exists(STATES_FILE):
    with open(STATES_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                seen_clips.add(json.loads(line)["clip_id"])
    print(f"Already saved states for {len(seen_clips)} clips.")

remaining = [p for p in all_pairs if p["pair_id"] not in processed_ids]
batch     = remaining[:SAMPLES_PER_RUN]

print(f"Progress : {len(processed_ids)}/{len(all_pairs)} "
      f"({100*len(processed_ids)/len(all_pairs):.1f}%)")
print(f"This run : {len(batch)} pairs")

if not batch:
    print("✅ All pairs processed!")
else:
    # ─────────────────────────────────────────────
    # MAIN LOOP
    # ─────────────────────────────────────────────
    errors = []

    with open(STATES_FILE, "a") as states_f:
        for sample in tqdm(batch, desc="CKA Omni"):
            pair_key = sample["style_pair"]
            try:
                f1 = get_features_and_output(sample["audio1_path"])
                f2 = get_features_and_output(sample["audio2_path"])

                # --- CKA ---
                for idx, (h1, h2) in enumerate(zip(f1["enc"], f2["enc"])):
                    global_results[pair_key][f"enc_{idx}"].append(
                        centered_kernel_alignment(h1, h2)
                    )
                if f1["prj"] is not None and f2["prj"] is not None:
                    global_results[pair_key]["projector"].append(
                        centered_kernel_alignment(f1["prj"], f2["prj"])
                    )
                for idx, (h1, h2) in enumerate(zip(f1["dec"], f2["dec"])):
                    global_results[pair_key][f"dec_{idx}"].append(
                        centered_kernel_alignment(h1, h2)
                    )

                # --- Probe states (deduplicated per clip) ---
                for audio_path, style, feats in [
                    (sample["audio1_path"], sample["style1"], f1),
                    (sample["audio2_path"], sample["style2"], f2),
                ]:
                    clip_id = os.path.basename(audio_path).replace(".wav", "")
                    if clip_id not in seen_clips:
                        record = {
                            "clip_id": clip_id,
                            "speaker": audio_path.split("/")[-4],
                            "emotion": style,
                            "utt_num": sample["sentence_id"],
                            "enc": [h.mean(axis=0).tolist() for h in feats["enc"]],
                            "prj": feats["prj"].mean(axis=0).tolist() if feats["prj"] is not None else None,
                            "dec": [h.mean(axis=0).tolist() for h in feats["dec"]],
                        }
                        states_f.write(json.dumps(record) + "\n")
                        states_f.flush()
                        seen_clips.add(clip_id)

                # --- Generations ---
                all_outputs.append({
                    "pair_id": sample["pair_id"],
                    "text":    sample["text"],
                    "style1":  sample["style1"],
                    "style2":  sample["style2"],
                    "output1": f1["output"],
                    "output2": f2["output"],
                })

                processed_ids.add(sample["pair_id"])
                save_all(processed_ids, global_results, all_outputs)

                del f1, f2
                torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                errors.append(f"pair_id={sample['pair_id']}: {e}")
                print(f"  ❌ {errors[-1]}")

    if errors:
        print(f"\n⚠️  {len(errors)} errors:")
        for e in errors[:5]:
            print(f"   {e}")

print(f"\n{'='*50}")
print(f"Progress : {len(processed_ids)}/{len(all_pairs)} "
      f"({100*len(processed_ids)/len(all_pairs):.1f}%)")
print(f"Remaining: {len(all_pairs) - len(processed_ids)} pairs")
print(f"Clips w/ states: {len(seen_clips)}")
print(f"Output   : {OUTPUT_DIR}/")
print(f"{'='*50}")
