# Paired manifests for CKA

Pre-built same-text, different-style/emotion pair lists for use with `run_cka.py --manifest`.

| File | Dataset | Description |
|------|---------|-------------|
| **expresso_local_2.json** | Expresso | Same sentence, different read-speech styles (e.g. confused vs default). Fields: `audio1_path`, `audio2_path`, `style1`, `style2`, `style_pair`, `text`. |
| **esp_local.json** | ESD | Same sentence, different emotions. Fields: `audio1_path`, `audio2_path`, `style1`, `style2`, `pair`. |

Paths in these files are absolute (from the machine where they were built). To use on another system, either place Expresso/ESD data at the same paths, or regenerate manifests with `build_expresso_all_manifest` / your ESD pair builder and adjust paths.

**Example:**

```bash
python -m colm.scripts.run_cka --manifest manifests/expresso_local_2.json --max-pairs 500 --output cka_expresso.npz
python -m colm.scripts.run_cka --manifest manifests/esp_local.json --max-pairs 500 --output cka_esd.npz
```
