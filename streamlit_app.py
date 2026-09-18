"""Phase 5 (Streamlit variant): normalize Thai text and classify sentiment.

Loads the fine-tuned model from the Hugging Face Hub (thitiwutsuk/thai-sentiment-wangchanberta)
so this app has no large local files to ship -- deployable on Streamlit Community Cloud's
free tier, which HF Spaces (Gradio/Docker) no longer offers without a PRO subscription.

Usage:
    streamlit run streamlit_app.py
"""

import os

import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.normalize import normalize_thai

MODEL_ID = os.environ.get("MODEL_ID", "thitiwutsuk/thai-sentiment-wangchanberta")
MAX_LENGTH = 64
LABEL_NAMES = ["pos", "neu", "neg", "q"]
LABEL_DISPLAY = {"pos": "Positive", "neu": "Neutral", "neg": "Negative", "q": "Question"}


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
st.title("Thai Text Normalization + Sentiment")
st.caption(
    "Normalizes informal Thai text (elongation, slang, emoji) and classifies sentiment "
    "with WangchanBERTa fine-tuned on the normalized output."
)

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
