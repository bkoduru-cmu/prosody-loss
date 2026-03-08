# Prosody in audio-language models: datasets and conversational model landscape

**Neither PersonaPlex nor the Seamless Interaction Dataset serves as a prosody-contrastive evaluation set comparable to ESD or Expresso**, but the broader Seamless ecosystem contains highly relevant resources. Meanwhile, among conversational speech-to-speech models, **Moshi, SpiRit-LM Expressive, and pGSLM offer the richest architectures for studying prosodic information flow**, with Moshi standing out as the most complete open-source system combining full-duplex dialogue, multi-codebook prosody representation, and accessible hidden states. Below is a detailed assessment of both datasets and all priority models, along with an analysis of prosody challenges unique to conversational settings.

---

## Part 1: PersonaPlex is a model, not a dataset

**PersonaPlex is NVIDIA's 7B-parameter real-time full-duplex speech-to-speech conversational model**, not a dataset. Published as "PersonaPlex: Voice and Role Control for Full Duplex Conversational Speech Models" (arXiv:2602.06053, accepted at ICASSP 2026), it was built by NVIDIA's ADLR team (Rajarshi Roy, Jonathan Raiman, et al.) on top of Kyutai's Moshi architecture with the Helium LLM backbone.

PersonaPlex's training data comprises **~3,467 hours** of mixed real and synthetic conversational audio — 1,217 hours from the Fisher English Corpus plus ~2,250 hours of synthetic dialogue generated via Qwen3-32B and GPT-OSS-120B, then vocalized using Chatterbox TTS. However, **this training data was not publicly released**. Only model weights (on HuggingFace at `nvidia/personaplex-7b-v1` under NVIDIA Open Model License) and inference code (GitHub, MIT license) are available.

**Suitability for prosody-contrastive CKA analysis: None.** PersonaPlex has no parallel same-text-different-emotion utterances, no emotion or prosody labels, and no downloadable audio corpus. It is English-only. It supports 18 voice embeddings for persona control, but this controls speaker identity, not emotional prosody. The training data is annotated with role/persona prompts (e.g., "You are a wise teacher"), not emotion categories. PersonaPlex is fundamentally incomparable to ESD or Expresso for the purpose of probing prosodic information loss through encoder/projector layers.

That said, PersonaPlex is potentially interesting as **a model to probe** rather than a dataset to use — its Moshi-based architecture with Inner Monologue provides a natural text bottleneck for studying information flow. Its "ServiceDuplexBench" evaluation benchmark is stated as upcoming but not yet released.

---

## Part 1: The Seamless ecosystem contains useful prosody resources

The term "Seamless" from Meta FAIR refers to two entirely separate projects that must be carefully distinguished.

**The Seamless Interaction Dataset (2025)** is an audiovisual face-to-face interaction video corpus — over **4,000 hours** of full-body dyadic conversation videos from 4,000+ participants (arXiv:2506.22554). It includes emotion detection annotations, facial action units, and body keypoints, but these are **visual/behavioral annotations, not speech prosody labels**. Available under CC-BY-NC 4.0 at `facebook/seamless-interaction` on HuggingFace. **Not suitable for speech prosody research** — it was designed for gesture and motion modeling.

**The Seamless Communication family (2023)**, specifically SeamlessExpressive (arXiv:2312.05187), is far more relevant. This speech translation system was trained using several datasets with prosodic properties:

**Expresso** (Meta, Interspeech 2023) is the most directly useful dataset in this ecosystem. It contains **~47 hours** of professional studio recordings (48kHz/24-bit) from **4 speakers** (2M, 2F) in **7 read speech styles** (default, confused, enunciated, happy, laughing, sad, whisper) plus 26 improvised dialogue styles. Critically, **the read speech section provides parallel same-text utterances across all 7 styles**, making it structurally analogous to ESD for CKA analysis. It also includes a unique **contrastive emphasis subset** where identical sentences are read 2–4 times with prosodic emphasis placed on different words — directly testing whether fine-grained prosodic focus survives the projector layer. Available at `ylacombe/expresso` on HuggingFace under CC-BY-NC 4.0.

| Feature | ESD | Expresso | Seamless Interaction |
|---|---|---|---|
| Parallel same-text utterances | ✅ 5 emotions × 10 speakers | ✅ 7 styles × 4 speakers | ❌ Not applicable |
| Emotion/style labels | 5 discrete emotions | 7 read styles + 26 improvised | Visual emotion only |
| Languages | English + Chinese | English only | English (video) |
| Total audio | ~29 hours | ~47 hours | 4,000+ hours (video) |
| Emphasis control | ❌ | ✅ Contrastive emphasis pairs | ❌ |
| CKA suitability | ★★★★★ | ★★★★☆ | ★☆☆☆☆ |

**mExpresso** extends Expresso's read speech into 6 languages (English, French, German, Italian, Mandarin, Spanish) with gender-matched bilingual speakers recording translations in the same 7 styles. Text translations are released via the `seamless_communication` repo (MIT), but **audio file availability remains unclear** — it was promised for release in late 2023 but status is uncertain as of early 2026. If available, mExpresso would enable **cross-lingual prosody-contrastive CKA analysis**.

**SeamlessAlignExpressive** offers 11,000+ hours of automatically aligned multilingual audio with expressivity-matched translation pairs, but it lacks discrete emotion labels and same-text-different-emotion structure, making it unsuitable for controlled CKA experiments. The SeamlessExpressive evaluation toolkit (in `stopes`) includes prosodic metrics like AutoPCP (prosodic consistency) and rate/pause scores that could supplement CKA analysis.

**Bottom line for datasets**: ESD remains the gold standard for emotion-contrastive CKA analysis. Expresso is a strong complement offering broader prosodic variation (7 styles + emphasis) with fewer speakers. The two are complementary: ESD for emotion-level analysis, Expresso for prosodic style and emphasis analysis. Neither PersonaPlex nor Seamless Interaction is suitable.

---

## Part 2: Moshi leads the conversational model landscape for prosody research

### Moshi — the premier architecture for prosodic information flow analysis

**Moshi** (Kyutai Labs, arXiv:2410.00037) is the strongest candidate for extending prosody research from ASR-oriented models to conversational ones. Its architecture differs fundamentally from the encoder→projector→LLM pipeline of Qwen2-Audio.

Moshi operates through three interconnected components. The **Mimi neural audio codec** (streaming, 12.5 Hz, 1.1 kbps) uses Split RVQ with 8 codebooks — a semantic first codebook distilled from WavLM embeddings plus 7 acoustic codebooks capturing timbre, pitch, and prosodic detail. The **Temporal Transformer** ("Helium," a custom **7B** text LLM) models temporal dependencies across two parallel audio streams (user + system) plus time-aligned text tokens (the "Inner Monologue"). The **Depth Transformer** maps from semantic tokens to the full 7-level acoustic RVQ at each time step.

This architecture is uniquely valuable for prosody research because it provides **three distinct information bottlenecks to probe**: (1) the semantic/acoustic split in Mimi's codebooks, where prosodic information distributes across levels; (2) the Temporal Transformer's hidden states, where you can measure how user prosody influences system responses; (3) the Inner Monologue text stream, which acts as an explicit text bottleneck analogous to the projector layer in Qwen2-Audio but architecturally more transparent. The model reportedly supports **70+ intonation patterns** and can modulate emotional expression.

Moshi is fully open-source (Apache 2.0 code, CC-BY 4.0 weights), with PyTorch, MLX, and Rust implementations. All hidden states are accessible via standard PyTorch hooks. A dedicated fine-tuning repo exists at `github.com/kyutai-labs/moshi-finetune`. The 7B backbone in bf16 requires ~14GB for weights, making **fine-tuning feasible on A100 80GB** with LoRA/QLoRA and gradient checkpointing. Streaming latency is **~200ms** on an L4 GPU.

### Models with explicit prosody representations

**pGSLM** (Meta/FAIR, ACL 2022, arXiv:2109.03264) is the foundational work demonstrating that explicit prosody token streams improve spoken language modeling. Its Multi-Stream Transformer Language Model (MS-TLM) autoregressively generates three parallel streams per time step: **HuBERT-derived speech units, discretized F0 (pitch), and duration tokens**. The key finding — that standard discrete speech units from HuBERT discard most prosodic information and that adding F0 + duration streams recovers it — directly motivates the CKA analysis of prosody loss. Trained on LibriSpeech read speech. Open-source under MIT via fairseq (`pytorch/fairseq/tree/main/examples/textless_nlp/pgslm`). Not conversational or streaming, but serves as the **theoretical baseline** for prosody preservation research.

**SpiRit-LM** (Meta/FAIR, TACL 2025, arXiv:2402.05755) extends this concept to a 7B LLaMA-based model with interleaved spoken and written tokens. The **Expressive version** uses HuBERT phonetic units alongside **explicit pitch tokens and style tokens**, demonstrating that adding prosodic token streams to a text LLM improves both semantic and expressive capabilities. Open-source (weights and inference code released). Not streaming, but highly relevant for CKA comparison — it shows how explicit prosody representations change layer-wise information structure.

**dGSLM** (Meta/FAIR, TACL 2023, arXiv:2203.16502) uses a dual-tower transformer with **cross-attention** between two speaker streams to model conversational dynamics. Trained on ~2,000 hours of Fisher conversational audio. Unlike pGSLM, it does **not** use explicit prosody tokens — relying solely on HuBERT units, which the authors acknowledge "discard most prosodic information." However, its cross-attention mechanism implicitly captures turn-taking, laughter, and backchannels. Open-source via fairseq. Valuable as a **prosody-poor conversational baseline** for CKA comparison.

### Models with encoder→projector→LLM architecture (directly comparable to Qwen2-Audio)

**LLaMA-Omni** (ICLR 2025, arXiv:2409.06666) uses a pipeline most directly comparable to Qwen2-Audio: **Whisper-large-v3 encoder (frozen, 632M) → 5× downsampling MLP adapter → Llama-3.1-8B-Instruct → NAR CTC speech decoder (425M) → HiFi-GAN vocoder**. Total ~9B parameters. The speech decoder explicitly takes LLM output hidden states as input, making it architecturally transparent for probing. The authors acknowledge that "the model currently provides speech output in a uniform voice, lacking control over paralinguistic qualities such as emotion, prosody, or dialect" — confirming prosody loss. Available on HuggingFace (ICTNLP/Llama-3.1-8B-Omni, academic use only). **LLaMA-Omni 2** (ACL 2025) extends to Qwen2.5-Instruct backbones from 0.5B–14B, enabling scaling studies of prosody loss. Fine-tuning feasible on A100 80GB with LoRA. Streaming latency ~226ms.

**Mini-Omni** (arXiv:2408.16725) and **Mini-Omni2** (arXiv:2410.11190) use **Whisper-small encoder → LlamaMLP adapter → Qwen2-0.5B → SNAC decoder**. At ~0.8B total parameters, these are lightweight baselines with the same projector architecture as Qwen2-Audio. Mini-Omni2 adds CLIP visual encoding and an interruption mechanism. Prosody preservation is acknowledged as "basic." Open-source on GitHub (`gpt-omni/mini-omni`). Trivially fits on A100 80GB. Useful for CKA analysis as a **minimal architecture control**.

**Freeze-Omni** (ICML 2025, arXiv:2411.00774) provides a unique experimental setup: a **CNN streaming encoder → alignment adapter → completely frozen Qwen2-7B-Instruct → AR single-codebook decoder**. The LLM backbone is never fine-tuned on speech in any training stage — only the encoder, adapter, and decoder are trained. This makes it an exceptional **control condition** for studying how prosodic information flows through a text-only pretrained LLM that has never seen audio data. The frozen LLM's hidden states reveal how much prosodic information a pure text model can incidentally capture. Available on GitHub (both `VITA-MLLM/Freeze-Omni` and `Tencent/Freeze-Omni`). Fits A100 80GB since only trainable components need gradients.

### Additional conversational models

**GLM-4-Voice** (Zhipu AI, arXiv:2412.02612, **9B**) uses a Whisper-based VQ tokenizer + GLM-4-9B backbone + CosyVoice flow-matching decoder. Notably, it supports **user-instructed prosody control** — emotion, speech rate, and dialect can be adjusted via spoken instructions. Interleaved text-speech generation preserves more information than cascade systems. Open-source at `github.com/zai-org/GLM-4-Voice`. Single codebook at 12.5 Hz is compressed but the decoder-stage expressivity partially compensates. **Moderate-high prosody relevance**.

**VITA-Audio** (NeurIPS 2025, arXiv:2505.03739) uses Qwen2.5-7B-Instruct with a Multiple Cross-modal Token Prediction (MCTP) module and GLM-4-Voice tokenizer/decoder. Achieves 3–5× inference speedup via interleaved text-audio generation. Open-source at `github.com/VITA-MLLM/VITA-Audio`. Earlier **VITA-1.0/1.5** used Mixtral-8x7B (~47B sparse), too large for single A100.

**OmniFlatten** (ACL 2025, arXiv:2410.17799) achieves full-duplex dialogue by "flattening" — interleaving user and assistant speech/text streams into a single sequence processable by an unmodified GPT backbone. Uses CosyVoice tokenizer (single semantic codebook). Three-stage training progresses from modality alignment through half-duplex to full-duplex dialogue. Code/weights availability not fully confirmed — demo at `omniflatten.github.io`. Addresses turn-taking and backchannels structurally but **does not explicitly model prosodic features**.

**Ichigo** (Jan.ai, arXiv:2410.15316, **~8B** on LLaMA 3.1) uses early-fusion mixed-modal architecture where discrete speech tokens from a custom Whisper-VQ tokenizer (22M params) are treated as first-class vocabulary tokens alongside text. No separate adapter or projector — speech and text tokens share the same embedding space. Open-source at `github.com/janhq/ichigo`. However, **speech output is not native** (requires external TTS), and prosodic information is likely lost in the VQ tokenization. Low prosody relevance but interesting architecturally for its projector-free design.

### Comprehensive model comparison

| Model | Params | Prosody representation | Full-duplex | Streaming | Open-source | A100 80GB | Prosody relevance |
|---|---|---|---|---|---|---|---|
| **Moshi** | 7B+ | Rich (8-codebook RVQ) | ✅ | ✅ 200ms | Apache 2.0 / CC-BY 4.0 | ✅ (LoRA) | ★★★★★ |
| **SpiRit-LM Exp.** | 7B | Explicit (pitch + style tokens) | ❌ | ❌ | ✅ Weights released | ✅ | ★★★★★ |
| **pGSLM** | ~100M | Explicit (F0 + duration streams) | ❌ | ❌ | ✅ MIT (fairseq) | ✅ | ★★★★★ |
| **LLaMA-Omni** | 9B | None (uniform voice) | ❌ | ✅ 226ms | Academic only | ✅ (LoRA) | ★★★★☆ |
| **Freeze-Omni** | 7–8B | Minimal (frozen LLM) | ✅ | ✅ | ✅ GitHub | ✅ | ★★★★☆ |
| **GLM-4-Voice** | 9B | Moderate (instructed control) | ❌ | ✅ | ✅ GitHub | ✅ (LoRA) | ★★★★☆ |
| **dGSLM** | ~100M | Implicit (HuBERT units only) | ✅ (offline) | ❌ | ✅ MIT (fairseq) | ✅ | ★★★☆☆ |
| **VITA-Audio** | 7B | Implicit (GLM-4 tokens) | ❌ | ✅ | ✅ GitHub | ✅ | ★★★☆☆ |
| **Mini-Omni/2** | 0.8–0.9B | Weak (basic SNAC) | Partial | ✅ | ✅ GitHub | ✅ (full FT) | ★★★☆☆ |
| **OmniFlatten** | LLM-scale | Implicit (semantic tokens) | ✅ | ✅ | Partial | Likely ✅ | ★★☆☆☆ |
| **Ichigo** | 8B | Minimal (WhisperVQ) | ❌ | ✅ 111ms | ✅ GitHub | ✅ | ★★☆☆☆ |

---

## Prosody challenges unique to conversational speech models

Conversational speech models face prosodic demands fundamentally different from ASR-oriented models like Qwen2-Audio or Whisper. These challenges have direct implications for CKA-based analysis of where prosodic information is encoded and lost.

### Turn-taking requires prosodic anticipation, not just recognition

Humans signal turn boundaries through **falling pitch, final lengthening, reduced intensity, and syntactic completion** — not silence alone. Research by Ekstedt and Skantze (2022) using Voice Activity Projection (VAP) models demonstrated that **prosody substantially improves turn-taking prediction**, and that removing prosodic variation (flattening F0, normalizing energy) degrades prediction accuracy. Their systematic perturbation analysis (SIGdial 2022, "How Much Does Prosody Help Turn-Taking?") confirmed that while lexical/syntactic cues also contribute, prosodic cues are critical for anticipating turn transitions. Multilingual studies (2024) show these cues are language-dependent — Japanese relies more on linguistic cues while English depends more on prosodic variation. For CKA analysis, **layers encoding turn-boundary prosodic cues will show distinctive representational structure** compared to mid-utterance representations. Models using discrete semantic tokens (dGSLM, OmniFlatten) likely lose these cues in tokenization, while Moshi's multi-codebook approach may preserve them.

### Backchannel generation is prosodically triggered

Backchannels ("uh-huh," "mm-hmm") are **prosodically driven, not lexically determined** — their timing, form, and function depend on the interlocutor's prosodic cues. Ward and Tsukahara (2000) established that low-pitch regions in the speaker's speech trigger backchannel responses. Recent work combines acoustic features with LLM representations for improved prediction (Wang et al., ICASSP 2024), and multilingual studies (Inoue et al., IWSDS 2026) show that Chinese backchannels are more sensitive to prosodic variation than Japanese ones. Most current models treat backchannels as regular speech tokens, but they cannot distinguish between a rising-pitch "mm-hmm" (encouragement) and a falling-pitch one (acknowledgment) without fine-grained prosodic encoding.

### No current model explicitly captures prosodic entrainment

**Prosodic entrainment** — the convergence of pitch, rate, and rhythm between conversational partners over time — is well-documented in human conversation and correlates with rapport and communicative success (Levitan & Hirschberg, 2011–2012). An open-source Entrainment-Metrics toolkit (IBERAMIA 2024) now enables quantification across multiple dimensions (proximity, convergence, synchrony). Yet **no current speech-to-speech model explicitly models entrainment**. dGSLM's cross-attention mechanism could theoretically produce entrainment-like behavior through mutual speaker influence, but this has not been evaluated. For CKA analysis, entrainment creates a **temporal dependency across speaker turns** that differs fundamentally from within-utterance prosody — models without multi-turn prosodic memory will fail to capture it.

### Streaming creates a fundamental tension with prosodic planning

Human speakers plan prosodic contours over entire phrases — sentence-level intonation (declarative falling, interrogative rising) requires knowing discourse function before utterance completion. Streaming models generate token-by-token with causal masking, making coherent prosodic planning difficult. Moshi's 160ms frame size limits prosodic look-ahead; GLM-4-Voice can begin synthesis with just 10 tokens (~0.8 seconds of context). For CKA analysis, **comparing representations at different context lengths would reveal streaming-specific prosody degradation patterns** distinct from the projector-layer loss observed in offline models like Qwen2-Audio.

---

## Strategic recommendations for extending CKA prosody analysis

The most productive path for extending your Qwen2-Audio projector-layer CKA analysis to conversational models involves a three-tier comparison. **Tier 1 (highest priority)**: Moshi, which provides a fundamentally different architecture where prosody distributes across Mimi's 8 codebook levels rather than passing through a single projector — enabling CKA measurement of where prosodic information lives in semantic vs. acoustic codebooks and how the Temporal Transformer processes it. **Tier 2**: LLaMA-Omni and Freeze-Omni, which share Qwen2-Audio's Whisper→adapter→LLM architecture but differ in whether the LLM is fine-tuned (LLaMA-Omni) or frozen (Freeze-Omni) on speech, creating a natural ablation for measuring how speech fine-tuning affects prosody encoding in LLM layers. **Tier 3**: pGSLM and SpiRit-LM Expressive as theoretical upper bounds, since their explicit prosody token streams guarantee prosodic information preservation — CKA between their layers and the prosody-agnostic models would quantify the representational gap.

For evaluation data, **ESD remains optimal** for emotion-contrastive analysis, with **Expresso as a complementary resource** offering broader prosodic style variation and the unique contrastive emphasis subset. The mExpresso multilingual extension, if audio is accessible, would enable cross-lingual prosody-contrastive studies.