"""Phase 5: Gradio demo -- normalize Thai text and classify sentiment.

Usage:
    python app.py
"""

import difflib
import os

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.normalize import normalize_thai

MODEL_DIR = os.environ.get("MODEL_DIR", "models/wangchanberta-normalized")
MAX_LENGTH = 64
LABEL_NAMES = ["pos", "neu", "neg", "q"]
LABEL_DISPLAY = {"pos": "Positive", "neu": "Neutral", "neg": "Negative", "q": "Question"}

if not os.path.isdir(MODEL_DIR):
    raise FileNotFoundError(
        f"No model found at '{MODEL_DIR}'. Train and save one first:\n"
        f"  python -m scripts.train_sentiment --variant normalized --save-model-dir {MODEL_DIR}"
    )

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()


def diff_highlight(original: str, normalized: str) -> list[tuple[str, str | None]]:
    """Character-level diff between original and normalized text, for gr.HighlightedText."""
    matcher = difflib.SequenceMatcher(None, original, normalized)
    segments = []
    for tag, _, _, j1, j2 in matcher.get_opcodes():
        chunk = normalized[j1:j2]
        if chunk:
            segments.append((chunk, "changed" if tag != "equal" else None))
    return segments


def predict(text: str):
    if not text or not text.strip():
        return {}, [], "_enter some text above_"

    result = normalize_thai(text)
    normalized_text = result["normalized_text"]

    inputs = tokenizer(
        normalized_text, truncation=True, max_length=MAX_LENGTH, padding="max_length", return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0].tolist()
    scores = {LABEL_DISPLAY[name]: probs[i] for i, name in enumerate(LABEL_NAMES)}

    highlighted = diff_highlight(text, normalized_text)

    if result["corrections"]:
        corrections_md = "\n".join(
            f"- **{c['type']}**: `{c['original']}` → `{c['normalized']}`" for c in result["corrections"]
        )
    else:
        corrections_md = "_no corrections made_"

    return scores, highlighted, corrections_md


with gr.Blocks(title="Thai Text Normalization + Sentiment") as demo:
    gr.Markdown(
        "# Thai Text Normalization + Sentiment\n"
        "Normalizes informal Thai text (elongation, slang, emoji) and classifies sentiment with "
        "WangchanBERTa fine-tuned on the normalized output. Highlighted spans below show what "
        "`normalize_thai()` changed."
    )

    input_box = gr.Textbox(
        label="Input text", placeholder="พิมพ์ข้อความภาษาไทย เช่น รีวิว/คอมเมนต์...", lines=3
    )
    submit_btn = gr.Button("Analyze", variant="primary")

    with gr.Row():
        with gr.Column():
            highlighted_out = gr.HighlightedText(
                label="Normalized text (highlighted = changed)",
                combine_adjacent=True,
                color_map={"changed": "orange"},
            )
            corrections_out = gr.Markdown(label="Corrections")
        with gr.Column():
            sentiment_out = gr.Label(label="Sentiment", num_top_classes=4)

    submit_btn.click(predict, inputs=input_box, outputs=[sentiment_out, highlighted_out, corrections_out])
    input_box.submit(predict, inputs=input_box, outputs=[sentiment_out, highlighted_out, corrections_out])

    gr.Examples(
        examples=[
            "อาหารช้ามากกกกก มั้ยอ่ะ เค้าไม่ชอบ 😭😭😭",
            "ของมาไวมากกก แพ็คดีสุดๆ ประทับใจค่ะ",
            "อยากรู้ว่าเปิดกี่โมงคะ",
        ],
        inputs=input_box,
    )

if __name__ == "__main__":
    demo.launch()
