# Thai Text Normalization

![Python](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)
![PyThaiNLP](https://img.shields.io/badge/NLP-PyThaiNLP-orange)
![Transformers](https://img.shields.io/badge/model-WangchanBERTa-yellow?logo=huggingface&logoColor=white)
![PyTorch](https://img.shields.io/badge/backend-PyTorch%20%2F%20MPS-EE4C2C?logo=pytorch&logoColor=white)
![Gradio](https://img.shields.io/badge/demo-Gradio-F97316?logo=gradio&logoColor=white)
![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)
![Status](https://img.shields.io/badge/status-in%20progress-blue)

Normalizes informal Thai social text (elongation, slang, emoji) into a standard form, to measure how much it improves downstream sentiment classification accuracy.

## Contents

- [Problem](#problem)
- [Pipeline](#pipeline)
- [Normalization Core](#normalization-core)
- [Datasets](#datasets)
- [Results](#results)
- [Module Reference](#module-reference)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Roadmap](#roadmap)

## Problem

- Thai social text (comments, reviews, chats) breaks standard NLP models trained on formal text
  - Elongation: `อร่อยยยยย` (soooo delicious)
  - Slang/abbreviations: `ทามมาย` (ทำไม), `ชิมิ` (ใช่ไหม), `555` (laughter)
  - Emoji/symbols: 😭, T_T, 😂
- Tokenizers make it worse: `word_tokenize("อร่อยยยยชิมิ")` splits into `['อร่อย', 'ยยย', 'ชิ', 'มิ']` — elongation has to be fixed *before* tokenizing, not after
- **Goal:** build `normalize_thai()` and prove it raises sentiment-classification accuracy vs. raw text

## Pipeline

- **Clean** ([src/cleaning.py](src/cleaning.py)) — strip URLs, mentions, hashtag symbols, extra whitespace
- **Reduce elongation** ([src/elongation.py](src/elongation.py)) — collapse repeated characters, character-level, before tokenizing
- **Map emoji/emoticons** ([src/emoji_map.py](src/emoji_map.py)) — emoji/text-emoticons → sentiment tokens
- **Tokenize** ([src/tokenization.py](src/tokenization.py)) — PyThaiNLP `word_tokenize` (`newmm` engine)
- **Slang lookup** ([src/slang_dict.py](src/slang_dict.py)) — per-token dictionary correction, sourced from MultiLexNorm++
- **Classify** ([scripts/train_sentiment.py](scripts/train_sentiment.py)) — fine-tune WangchanBERTa on Wisesight Sentiment, raw vs. normalized text
- **Evaluate** ([scripts/evaluate_results.py](scripts/evaluate_results.py)) — compare accuracy/F1/confusion matrices, curate cases normalization fixed
- **Demo** *(planned)* — Gradio app deployed to Hugging Face Spaces

## Normalization Core

`normalize_thai(text)` ([src/normalize.py](src/normalize.py)) runs the steps above in order and returns a dict:

```python
>>> normalize_thai("อาหารช้ามากกกกก มั้ยอ่ะ เค้าไม่ชอบ 😭😭😭")
{
  "text": "อาหารช้ามากกกกก มั้ยอ่ะ เค้าไม่ชอบ 😭😭😭",
  "normalized_text": "อาหาร ช้า มาก ไหม อะ เขา ไม่ ชอบ [neg_emoji] [neg_emoji] [neg_emoji]",
  "tokens": ["อาหาร", "ช้า", "มาก", "ไหม", "อะ", "เขา", "ไม่", "ชอบ", "[neg_emoji]", ...],
  "corrections": [
    {"original": "...มากกกกก...", "normalized": "...มาก...", "type": "elongation"},
    {"original": "...😭😭😭",      "normalized": "...[neg_emoji]...", "type": "emoji"},
    {"original": "มั้ย", "normalized": "ไหม", "type": "slang_dict"},
    {"original": "อ่ะ",  "normalized": "อะ",  "type": "slang_dict"},
    {"original": "เค้า", "normalized": "เขา", "type": "slang_dict"}
  ]
}
```

<details>
<summary>Design notes / known gaps</summary>

- Elongation reduction skips digit runs (`555` = laughter, not a typo) and emoji runs — only collapses letters/punctuation
- The slang dictionary only corrects entries seen ≥3 times in MultiLexNorm++'s training data (17k+ entries) — coverage gaps remain (e.g. `ชิมิ` isn't in it, so it still gets mis-tokenized)
- Emoji/text-emoticons map to one of 3 tags: `[pos_emoji]`, `[neg_emoji]`, `[emoji]` (unrecognized) — a coarse sentiment signal, not per-emotion granularity

</details>

## Datasets

| Dataset | Purpose | Notes |
|---|---|---|
| [`hadung1802/mlnorm-resources`](https://huggingface.co/datasets/hadung1802/mlnorm-resources) | Slang/lexical normalization pairs (Thai) | Not a standard `load_dataset` repo — copied locally to [data/raw/mlnorm/](data/raw/mlnorm/) by `scripts/fetch_data.py`
| [`pythainlp/wisesight_sentiment`](https://huggingface.co/datasets/pythainlp/wisesight_sentiment) | Sentiment labels (pos/neu/neg/q) | Primary dataset for measuring accuracy before/after normalization — copied locally to [data/raw/wisesight/](data/raw/wisesight/)
| [`Wongnai/wongnai_reviews`](https://huggingface.co/datasets/Wongnai/wongnai_reviews) | Rating classification (1–5 stars) | Not used — reviews are long-form (~550 chars avg) and skew to 3–4 stars, outside this project's short-text scope

<details>
<summary>Known dataset quirks (see <code>scripts/explore_datasets.py</code>)</summary>

- MultiLexNorm++ `test.norm` is a **blind test set** (labels blanked, 0/34,167 non-empty) — use `dev.norm` for evaluation instead
- MultiLexNorm++ `train.norm`/`dev.norm` correct ~4.7% of tokens each — mostly elongation, spacing (`ๆ`), and informal spellings (`เค้า`→`เขา`, `มั้ย`→`ไหม`)
- Wisesight labels are imbalanced (`neu` ~55%, `neg` ~25%, `pos` ~18%, `q` ~2%) — report macro-F1, not just accuracy
- Wongnai ratings skew heavily to 3–4 stars (train: 1★=415 vs. 4★=18,770) — would need bucketing into neg/neu/pos if ever used

</details>

## Results

Phase 3/4 fine-tunes `airesearch/wangchanberta-base-att-spm-uncased` on Wisesight Sentiment (3 epochs, batch 32, max length 64), once on raw text and once on `normalize_thai()` output, with everything else held constant — isolating normalization as the only variable.

| Setup | Accuracy | Macro-F1 | pos F1 | neu F1 | neg F1 | q F1 |
|---|---|---|---|---|---|---|
| Raw text | 74.35% | 64.28% | 49.7% | 79.4% | 79.0% | 49.0% |
| Normalized text | 74.32% | **65.51%** | **52.9%** | 79.0% | 79.2% | **50.9%** |

**Takeaway:** accuracy barely moves (dominated by the majority `neu`/`neg` classes, which don't need normalization to be classified well), but **macro-F1 improves by +1.2pp**, driven almost entirely by the minority classes — `pos` +3.2pp, `q` +1.9pp. This matches the hypothesis: normalization helps most where the model has the fewest examples to fall back on, since it can no longer lean on memorized surface forms and has to generalize from the cleaned-up token instead.

<details>
<summary>Why q/pos lag behind neu/neg in absolute terms</summary>

Wisesight's label distribution is skewed (`neu` 55%, `q` only 2% of training data), so the minority classes get far fewer training examples — expected under plain cross-entropy training with no class weighting. This affects both the raw and normalized runs, so the *relative* raw-vs-normalized comparison stays valid even though absolute per-class numbers are uneven.

</details>

Run `python -m scripts.evaluate_results` for the full comparison table, confusion matrices, and curated examples of cases normalization fixed.

## Module Reference

| Module | Responsibility |
|---|---|
| [src/cleaning.py](src/cleaning.py) | Strip URLs/mentions/hashtag symbols, collapse whitespace |
| [src/elongation.py](src/elongation.py) | Collapse repeated characters (`มากกกกก` → `มาก`) |
| [src/emoji_map.py](src/emoji_map.py) | Emoji + text-emoticons → `[pos_emoji]`/`[neg_emoji]`/`[emoji]` |
| [src/tokenization.py](src/tokenization.py) | PyThaiNLP word tokenization |
| [src/slang_dict.py](src/slang_dict.py) | Token-level slang → standard-form lookup (MultiLexNorm++) |
| [src/normalize.py](src/normalize.py) | Combines all of the above into `normalize_thai()` |

## Project Structure

```
src/            reusable normalize_thai() pipeline code
scripts/        data-fetching, exploration, training, and evaluation scripts
tests/          pytest unit tests (one file per src/ module)
data/raw/       raw dataset files, fetched via scripts/fetch_data.py (git-tracked)
results/        saved metrics, predictions, and training logs (git-tracked)
thai-text-normalization-plan-en.md   full project plan & milestones
```

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m scripts.fetch_data   # copies the raw dataset files into data/raw/
```

## Usage

- Fetch raw dataset files into `data/raw/` (run once after setup)
  ```bash
  python -m scripts.fetch_data
  ```
- Explore dataset structure
  ```bash
  python scripts/explore_datasets.py
  ```
- Run the Phase 1 cleaning + tokenization demo on real Wisesight samples
  ```bash
  python -m scripts.phase1_demo
  ```
- Fine-tune WangchanBERTa on raw or normalized text
  ```bash
  python -m scripts.train_sentiment --variant raw
  python -m scripts.train_sentiment --variant normalized
  ```
  - `--limit N` truncates each split, for a fast smoke test before a full run
  - On Apple Silicon, set `PYTORCH_ENABLE_MPS_FALLBACK=1` — some ops can otherwise stall on the MPS backend
- Compare raw vs. normalized results once both have run
  ```bash
  python -m scripts.evaluate_results
  ```
- Run the test suite
  ```bash
  python -m pytest -q
  ```

## Roadmap

- [x] Phase 0 — repo/env setup, dataset exploration
- [x] Phase 1 — text cleaning & tokenization
- [x] Phase 2 — `normalize_thai()` core (elongation, slang dictionary, emoji mapping)
- [x] Phase 3 — WangchanBERTa fine-tuning, raw vs. normalized
- [ ] Phase 4 — evaluation & error analysis
- [ ] Phase 5 — Gradio demo, deployed to Hugging Face Spaces
- [ ] Phase 6 — documentation & report

See [thai-text-normalization-plan-en.md](thai-text-normalization-plan-en.md) for full detail.
