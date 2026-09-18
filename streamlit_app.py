"""Phase 5 (Streamlit variant): project report + live demo, normalize Thai text
and classify sentiment.

Loads the fine-tuned model from the Hugging Face Hub (thitiwutsuk/thai-sentiment-wangchanberta)
so this app has no large local files to ship -- deployable on Streamlit Community Cloud's
free tier, which HF Spaces (Gradio/Docker) no longer offers without a PRO subscription.

Usage:
    streamlit run streamlit_app.py
"""

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.normalize import normalize_thai

MODEL_ID = os.environ.get("MODEL_ID", "thitiwutsuk/thai-sentiment-wangchanberta")
MAX_LENGTH = 64
LABEL_NAMES = ["pos", "neu", "neg", "q"]
LABEL_DISPLAY = {"pos": "Positive", "neu": "Neutral", "neg": "Negative", "q": "Question"}
RESULTS_DIR = Path(__file__).parent / "results"


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()
    return tokenizer, model


def predict(text: str):
    tokenizer, model = load_model()
    result = normalize_thai(text)
    normalized_text = result["normalized_text"]

    inputs = tokenizer(
        normalized_text, truncation=True, max_length=MAX_LENGTH, padding="max_length", return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0].tolist()
    scores = {LABEL_DISPLAY[name]: probs[i] for i, name in enumerate(LABEL_NAMES)}
    return result, scores


@st.cache_data
def load_benchmark():
    """Everything the report section needs, computed once and cached."""
    raw_metrics = json.load(open(RESULTS_DIR / "raw_metrics.json", encoding="utf-8"))
    norm_metrics = json.load(open(RESULTS_DIR / "normalized_metrics.json", encoding="utf-8"))

    def load_predictions(variant):
        rows = []
        with open(RESULTS_DIR / f"{variant}_predictions.jsonl", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        return rows

    raw_preds = load_predictions("raw")
    norm_preds = load_predictions("normalized")

    fixed, regressed = [], []
    for r, n in zip(raw_preds, norm_preds):
        if not r["correct"] and n["correct"]:
            fixed.append({"raw": r["text"], "normalized": n["text"], "true": r["true_label"], "raw_pred": r["pred_label"], "norm_pred": n["pred_label"]})
        elif r["correct"] and not n["correct"]:
            regressed.append({"raw": r["text"], "normalized": n["text"], "true": r["true_label"], "raw_pred": r["pred_label"], "norm_pred": n["pred_label"]})

    fixed_with_correction = [c for c in fixed if normalize_thai(c["raw"])["corrections"]]

    return {
        "raw_metrics": raw_metrics,
        "norm_metrics": norm_metrics,
        "fixed": fixed,
        "regressed": regressed,
        "fixed_with_correction": fixed_with_correction,
    }


def confusion_df(cm: list[list[int]]) -> pd.DataFrame:
    labels = [LABEL_DISPLAY[l] for l in LABEL_NAMES]
    df = pd.DataFrame(cm, index=[f"true: {l}" for l in labels], columns=[f"pred: {l}" for l in labels])
    return df


st.set_page_config(page_title="Thai Text Normalization + Sentiment", page_icon="🇹🇭", layout="centered")

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

st.title("Thai Text Normalization + Sentiment")
st.caption(
    "A full project report — problem, method, results, and what was learned — "
    "followed by a live demo you can try yourself at the bottom."
)

st.header("1. Problem")
st.markdown(
    """
Thai social text (comments, reviews, chats) breaks standard NLP models trained on formal text:

- **Elongation** — `อร่อยยยยย` (soooo delicious)
- **Slang / abbreviations** — `ทามมาย` (ทำไม), `ชิมิ` (ใช่ไหม), `555` (laughter)
- **Emoji / symbols** — 😭, T_T, 😂

It gets worse once you tokenize: PyThaiNLP's word-splitter turns `อร่อยยยยชิมิ` into
`['อร่อย', 'ยยย', 'ชิ', 'มิ']` — the elongation and the slang both get mangled into garbage
tokens instead of being recognized as words.

**Goal:** build a `normalize_thai()` function that cleans this up, and *measure* — not
assume — whether doing so actually improves sentiment classification.
"""
)

st.header("2. Datasets")
st.markdown(
    """
| Dataset | Used for | Key fact |
|---|---|---|
| **MultiLexNorm++** (Thai slice) | Source of the slang → standard-form dictionary | 17k+ mined entries; ~4.7% of tokens in the training data needed correction |
| **Wisesight Sentiment** | Training & evaluating the sentiment classifier | 21,628 / 2,404 / 2,671 train/val/test; labels are imbalanced (`neu` ~55%, `q` only ~2%) |
"""
)
st.caption("Because labels are imbalanced, **macro-F1** is tracked alongside accuracy — accuracy alone would hide how the minority classes perform.")

st.header("3. Method")
st.markdown(
    """
- **Normalize first, at the character level** — collapse elongation (`มากกกก` → `มาก`) and map
  emoji/emoticons to a sentiment tag (`😭` → `[neg_emoji]`), *before* tokenizing
- **Then correct at the token level** — look up each word in a slang dictionary mined from
  MultiLexNorm++'s annotated corpus (e.g. `มั้ย` → `ไหม`, `เค้า` → `เขา`)
- **Fine-tune WangchanBERTa** (a Thai RoBERTa model) on Wisesight Sentiment — twice, with
  everything (base model, hyperparameters, random seed) held identical except the input text:
  once on the raw `texts` column, once on `normalize_thai()`'s output
"""
)
with st.expander("Full training details"):
    st.markdown(
        """
        - New 4-class classification head on top of `airesearch/wangchanberta-base-att-spm-uncased`, randomly initialized
        - Loss: cross-entropy · Optimizer: AdamW · LR: 5e-5 with linear decay and 10% warmup
        - 3 epochs, batch size 32, max sequence length 64 tokens
        - Final metrics come from the **test** split, never seen during training or per-epoch validation
        """
    )

st.header("4. Results")

try:
    bench = load_benchmark()
    raw_metrics, norm_metrics = bench["raw_metrics"], bench["norm_metrics"]

    col1, col2 = st.columns(2)
    col1.metric(
        "Accuracy",
        f"{norm_metrics['accuracy']:.1%}",
        f"{(norm_metrics['accuracy'] - raw_metrics['accuracy']) * 100:+.1f}pp vs. raw",
    )
    col2.metric(
        "Macro-F1",
        f"{norm_metrics['macro_f1']:.1%}",
        f"{(norm_metrics['macro_f1'] - raw_metrics['macro_f1']) * 100:+.1f}pp vs. raw",
    )

    st.markdown("**Per-class F1**")
    per_class = pd.DataFrame(
        {
            "raw": [raw_metrics["per_class_f1"][l] for l in LABEL_NAMES],
            "normalized": [norm_metrics["per_class_f1"][l] for l in LABEL_NAMES],
        },
        index=[LABEL_DISPLAY[l] for l in LABEL_NAMES],
    )
    st.bar_chart(per_class)
    st.success(
        "Accuracy barely moves (dominated by the majority Neutral/Negative classes), but "
        "**macro-F1 improves by +1.2pp** — almost entirely from the minority classes: "
        "Positive +3.2pp, Question +1.9pp."
    )

    with st.expander("Confusion matrices (test set)"):
        c1, c2 = st.columns(2)
        c1.caption("Raw text")
        c1.dataframe(confusion_df(raw_metrics["confusion_matrix"]))
        c2.caption("Normalized text")
        c2.dataframe(confusion_df(norm_metrics["confusion_matrix"]))

    st.markdown("**What actually changed, prediction by prediction**")
    fixed, regressed, fixed_wc = bench["fixed"], bench["regressed"], bench["fixed_with_correction"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Fixed (wrong → correct)", len(fixed))
    m2.metric("Broken (correct → wrong)", len(regressed))
    m3.metric("Net change", len(fixed) - len(regressed))
    st.info(
        f"Almost as many predictions were fixed as broken ({len(fixed)} vs. {len(regressed)}) — "
        "that's *why* accuracy barely moves. But the two sets of flips land on different "
        "classes: fixes concentrate on the minority classes, regressions land on classes that "
        "had correct predictions to spare — a redistribution that helps macro-F1 without "
        f"changing raw accuracy. Only **{len(fixed_wc)}/{len(fixed)}** of the fixes came from an "
        "actual correction (elongation/emoji/slang) — the rest were fixed by tokenizer spacing "
        "alone, with no correction logged at all."
    )

    with st.expander(f"Example: a case normalization fixed"):
        if fixed_wc:
            c = fixed_wc[0]
            st.markdown(f"**Raw:** `{c['raw']}`")
            st.markdown(f"**Normalized:** `{c['normalized']}`")
            st.markdown(f"True label: `{c['true']}` — raw predicted `{c['raw_pred']}`, normalized predicted `{c['norm_pred']}` ✅")

    with st.expander(f"Example: a case normalization broke"):
        if regressed:
            c = regressed[0]
            st.markdown(f"**Raw:** `{c['raw']}`")
            st.markdown(f"**Normalized:** `{c['normalized']}`")
            st.markdown(f"True label: `{c['true']}` — raw predicted `{c['raw_pred']}` ✅, normalized predicted `{c['norm_pred']}` ❌")

except FileNotFoundError:
    st.info("Benchmark results not found in this deployment.")

st.header("5. Limitations")
st.markdown(
    """
- **Dictionary coverage is incomplete** — only slang seen ≥3 times in MultiLexNorm++'s corpus is
  corrected, so gaps remain (e.g. `ชิมิ` isn't in it and still gets mis-tokenized)
- **Some of the gain is a tokenizer-spacing artifact**, not a linguistic correction — see the
  fix breakdown above
- **Sarcasm and tone are out of scope** — `normalize_thai()` only fixes surface form, not meaning
  that depends on context
- **Emoji mapping is coarse** — all emoji collapse into 3 tags (positive/negative/neutral), not
  per-emotion granularity
- **Metrics are from a single training run**, not averaged over multiple seeds
"""
)

st.header("6. Conclusion")
st.markdown(
    """
Normalizing informal Thai text measurably improves **macro-F1** (+1.2pp) on Wisesight
Sentiment — concentrated in the classes with the fewest training examples — while leaving raw
accuracy roughly unchanged, since it fixes about as many predictions as it breaks. About a
third of the fixes trace back to tokenizer spacing rather than the normalization logic itself,
a distinction worth isolating in any follow-up work.
"""
)

st.divider()

# ---------------------------------------------------------------------------
# Live demo
# ---------------------------------------------------------------------------

st.header("7. Try it yourself")

examples = [
    "อาหารช้ามากกกกก มั้ยอ่ะ เค้าไม่ชอบ 😭😭😭",
    "ของมาไวมากกก แพ็คดีสุดๆ ประทับใจค่ะ",
    "อยากรู้ว่าเปิดกี่โมงคะ",
]
chosen_example = st.selectbox("Try an example, or type your own below:", [""] + examples)
text = st.text_area("Input text", value=chosen_example, height=100, placeholder="พิมพ์ข้อความภาษาไทย...")

if st.button("Analyze", type="primary") and text.strip():
    with st.spinner("Loading model / running inference..."):
        result, scores = predict(text)

    top_label = max(scores, key=scores.get)
    st.subheader(f"Sentiment: {top_label} ({scores[top_label]:.1%} confidence)")
    st.bar_chart(scores)

    st.subheader("Normalized text")
    st.write(result["normalized_text"])

    st.subheader("Corrections")
    if result["corrections"]:
        for c in result["corrections"]:
            st.markdown(f"- **{c['type']}**: `{c['original']}` → `{c['normalized']}`")
    else:
        st.write("_no corrections made_")
