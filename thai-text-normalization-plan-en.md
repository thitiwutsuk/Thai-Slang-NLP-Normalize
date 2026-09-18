# Thai Text Normalization Project

## Problem Statement

Thai text on social media (comments, reviews, chats) is often written in an informal style, including:
- Elongated words: "อร่อยยยยย" (soooo delicious), "ชอบบบบมาก" (looove it)
- Slang / abbreviations: "ทามมาย" (ทำไม = why), "ชิมิ" (ใช่ไหม = right?), "555" (laughter, like "haha")
- Emojis and emotion symbols: 😭, T_T, 😂

Standard NLP models (e.g. sentiment analysis, NER) are trained mainly on formal Thai text, which causes them to misinterpret this kind of text or miss important emotional signals. This hurts the accuracy of downstream tasks such as sentiment analysis, social listening, and customer feedback analysis.

**Goal:** Build a Thai Text Normalization system that converts informal text into a standardized form, in order to improve the accuracy of downstream NLP models (especially sentiment analysis), and to measure the improvement before vs. after normalization.

**Scope:**
- Focus on short-form text (social comments/reviews/posts), not formal documents
- Covers: elongation reduction, dictionary-based slang mapping, emoji/symbol tokenization
- Out of scope: full sarcasm detection (listed as future work), speech-to-text/TTS

**Success Metrics:**
- Sentiment classification accuracy improves significantly when using normalized text vs. raw text
- Slang dictionary covers at least [target number] of common informal terms
- A working demo is deployed and usable end-to-end

---

## Tech Stack

- **Language:** Python 3.9+
- **NLP:** PyThaiNLP (tokenization, normalize), regex, emoji
- **Data:** pandas, numpy, HuggingFace `datasets`
- **Model:** transformers (WangchanBERTa), PyTorch, scikit-learn
- **Demo:** Gradio
- **Deployment:** Hugging Face Spaces
- **Versioning:** Git/GitHub

---

## Datasets

| Dataset | Purpose | Source |
|---|---|---|
| MultiLexNorm++ (2026) | Slang/lexical normalization pairs (Thai) | `hadung1802/mlnorm-resources` (HuggingFace) |
| Wisesight Sentiment | Sentiment labels (pos/neg/neutral/question) | `github.com/PyThaiNLP/wisesight-sentiment` |
| Wongnai Reviews | Rating classification (1-5 stars), optional extension | HuggingFace |

---

## Work Plan / Milestones

### Phase 0: Setup
- [x] Create repo, virtualenv, requirements.txt
- [x] Load and explore the MultiLexNorm++ dataset structure
- [x] Load and explore the Wisesight Sentiment dataset

### Phase 1: Text Cleaning & Tokenization
- [x] Write a cleaning function (strip URLs, mentions, hashtag symbols, extra whitespace)
- [x] Write a tokenization function using PyThaiNLP `word_tokenize`
- [x] Write unit tests for edge cases (empty text, emoji-only, numbers-only)

### Phase 2: Normalization Core
- [x] Write an elongation-reduction function (regex to collapse repeated characters)
- [x] Build a dictionary lookup from MultiLexNorm++ (slang → standard mapping)
- [x] Write an emoji/symbol → sentiment-token mapping function
- [x] Combine everything into a single `normalize_thai(text) -> dict` function (returns normalized text + correction log)

### Phase 3: Sentiment Model Baseline & Fine-tuning
- [x] Train/evaluate a baseline model on raw text (fine-tune WangchanBERTa on Wisesight)
- [x] Train/evaluate the same model on normalized text
- [x] Compare accuracy, F1, and confusion matrix between the two setups

### Phase 4: Evaluation & Analysis
- [ ] Build tables/charts comparing before vs. after normalization
- [ ] Analyze error cases that normalization actually fixed (curate examples)
- [ ] Document limitations (e.g. slang not covered by the dictionary, sarcasm)

### Phase 5: Demo App
- [ ] Build a Gradio interface: input text box → normalized text (highlight corrections) + sentiment + confidence
- [ ] Deploy to Hugging Face Spaces

### Phase 6: Documentation
- [ ] Write README (problem, approach, results, how to run)
- [ ] Write a short report/slide summarizing results (for thesis/portfolio use)

---

## Expected Outputs

1. **Library/function** `normalize_thai()`, reusable across projects
2. **Dictionary file** (.csv/.json) mapping slang → standard Thai
3. **Evaluation results** comparing sentiment accuracy before vs. after normalization
4. **Deployed demo web app**

---

## Open Questions / Decisions Needed Before Starting
- What dictionary size is needed (start with a small manual list, or use the full MultiLexNorm++)?
- Use the full WangchanBERTa model or a smaller variant (for faster fine-tuning)?
- Is a real deployment required, or is a notebook-based demo sufficient?
