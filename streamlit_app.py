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


st.set_page_config(page_title="Thai Text Normalization + Sentiment", page_icon="🇹🇭")

# ---------------------------------------------------------------------------
# Report: Problem -> Approach -> Results (short, with a chart)
# ---------------------------------------------------------------------------

st.title("Thai Text Normalization + Sentiment")
st.caption("A short project report, followed by a live demo you can try yourself.")

st.header("1. Problem")
st.markdown(
    """
Thai social text (comments, reviews, chats) breaks standard NLP models trained on formal text:

- **Elongation** — `อร่อยยยยย` (soooo delicious)
- **Slang / abbreviations** — `ทามมาย` (ทำไม), `ชิมิ` (ใช่ไหม), `555` (laughter)
- **Emoji / symbols** — 😭, T_T, 😂

**Goal:** build `normalize_thai()` and measure whether it improves sentiment classification.
"""
)

st.header("2. Approach")
st.markdown(
    """
- **Normalize** — collapse elongation, map emoji to a sentiment tag, look up slang in a
  dictionary mined from MultiLexNorm++ — all *before* tokenizing, since elongation breaks the tokenizer otherwise
- **Classify** — fine-tune `WangchanBERTa` on Wisesight Sentiment, once on raw text and once on
  normalized text, with everything else (model, hyperparameters, seed) held identical
"""
)

st.header("3. Results")

try:
    raw_metrics = json.load(open(RESULTS_DIR / "raw_metrics.json", encoding="utf-8"))
    norm_metrics = json.load(open(RESULTS_DIR / "normalized_metrics.json", encoding="utf-8"))

    col1, col2 = st.columns(2)
    col1.metric("Accuracy", f"{norm_metrics['accuracy']:.1%}", f"{(norm_metrics['accuracy'] - raw_metrics['accuracy']) * 100:+.1f}pp vs. raw")
    col2.metric("Macro-F1", f"{norm_metrics['macro_f1']:.1%}", f"{(norm_metrics['macro_f1'] - raw_metrics['macro_f1']) * 100:+.1f}pp vs. raw")

    per_class = pd.DataFrame(
        {
            "raw": [raw_metrics["per_class_f1"][l] for l in LABEL_NAMES],
            "normalized": [norm_metrics["per_class_f1"][l] for l in LABEL_NAMES],
        },
        index=[LABEL_DISPLAY[l] for l in LABEL_NAMES],
    )
    st.bar_chart(per_class)
    st.caption(
        "Accuracy barely moves, but macro-F1 improves — driven by the minority classes "
        "(Positive, Question), where normalization helps most because the model has fewer "
        "examples to fall back on."
    )
except FileNotFoundError:
    st.info("Benchmark results not found in this deployment.")

st.divider()

# ---------------------------------------------------------------------------
# Live demo
# ---------------------------------------------------------------------------

st.header("4. Try it yourself")

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
