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


st.set_page_config(page_title="Thai Text Normalization + Sentiment", page_icon="🇹🇭", layout="wide")

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

st.title("Thai Text Normalization + Sentiment")
st.markdown("Report by [**thitiwutsuk**](https://github.com/thitiwutsuk/Thai-Slang-NLP-Normalize)")
st.caption(
    "A full project report — problem, method, results, and what was learned — "
    "followed by a live demo you can try yourself at the bottom."
)

st.markdown(
    "**Executive summary.** Informal Thai social text routinely defeats NLP models trained on "
    "formal language. This project built a rule-based normalizer and measured its effect on a "
    "fine-tuned sentiment classifier in a controlled experiment: normalization raised "
    f":orange[**macro-F1 by 1.2 percentage points**], concentrated almost entirely in the two "
    "minority sentiment classes, while overall accuracy stayed effectively flat."
)

st.header("Problem Statement")
st.markdown(
    """
Informal Thai text — social comments, product reviews, chat messages — routinely breaks NLP
models trained on formal, edited language. This matters commercially: sentiment analysis on
exactly this kind of text underpins social listening, customer-feedback triage, and review
moderation, so systematic misreadings translate directly into misclassified customer signal.
Three failure modes recur constantly:

- **Elongation** — `อร่อยยยยย` ("sooo delicious"), used for emphasis and stripped of meaning by most tokenizers
- **Slang and abbreviations** — `ทามมาย` (ทำไม, "why"), `ชิมิ` (ใช่ไหม, "right?"), `555` (a laughter marker)
- **Emoji and symbols** — 😭, T_T, 😂, carrying sentiment that plain-text models never see

Tokenization compounds the problem rather than absorbing it: PyThaiNLP's word-splitter turns
`อร่อยยยยชิมิ` into `['อร่อย', 'ยยย', 'ชิ', 'มิ']` — elongation and slang both fragment into
meaningless tokens instead of being recognized as words. In this project's own training data,
roughly :orange[**4.7% of tokens**] required correction of exactly this kind.

**Objective.** Build a `normalize_thai()` function that resolves these three failure modes, and
*measure* — rather than assume — whether doing so actually improves downstream sentiment
classification accuracy.
"""
)

st.header("Data Sources")
st.markdown(
    """
Two public, purpose-built datasets were selected: one to supply normalization knowledge, the
other to provide an independent benchmark for measuring its downstream effect.

| Dataset | Role | Key fact |
|---|---|---|
| **MultiLexNorm++** (Thai slice) | Source of the slang → standard-form dictionary | 17k+ mined entries; ~4.7% of tokens in the training data needed correction |
| **Wisesight Sentiment** | Training and evaluating the sentiment classifier | 21,628 / 2,404 / 2,671 train / validation / test; labels are imbalanced (`neu` ~55%, `q` only ~2%) |

Wisesight was chosen specifically because it is drawn from real, short-form Thai social media
posts rather than curated review text — the same register the normalization pipeline targets.
"""
)
st.caption("Because labels are imbalanced, **macro-F1** is tracked alongside accuracy throughout this report — accuracy alone would hide how the minority classes perform.")

st.header("Methodology")
st.markdown(
    """
The pipeline separates two concerns deliberately: normalization is a fixed, deterministic
preprocessing step, and its effect is isolated with a controlled experiment rather than inferred
from a single training run.

- **Normalize first, at the character level** — collapse elongation (`มากกกก` → `มาก`) and map
  emoji/emoticons to a sentiment tag (`😭` → `[neg_emoji]`), *before* tokenizing. Order matters:
  running the tokenizer on un-normalized text is what produces the garbage tokens described above.
- **Then correct at the token level** — look up each word in a slang dictionary mined from
  MultiLexNorm++'s annotated corpus (e.g. `มั้ย` → `ไหม`, `เค้า` → `เขา`)
- **Fine-tune WangchanBERTa** — a RoBERTa-architecture model pretrained specifically on Thai text,
  chosen over multilingual alternatives for its stronger native handling of Thai morphology and
  the absence of word boundaries. Trained twice on Wisesight Sentiment, with everything (base
  model, hyperparameters, random seed) held identical except the input text: once on the raw
  `texts` column, once on `normalize_thai()`'s output — isolating normalization as the one
  variable that changes between runs
"""
)

st.markdown("**`normalize_thai()` in action** — computed live, right now, by this app:")
_example_result = normalize_thai("อาหารช้ามากกกกก มั้ยอ่ะ เค้าไม่ชอบ 😭😭😭")
st.code(json.dumps(_example_result, ensure_ascii=False, indent=2), language="json")
with st.expander("Design notes / known gaps"):
    st.markdown(
        """
        - Elongation reduction skips digit runs (`555` = laughter, not a typo) and emoji runs — only collapses letters/punctuation
        - The slang dictionary only corrects entries seen ≥3 times in MultiLexNorm++'s training data (17k+ entries) — coverage gaps remain (e.g. `ชิมิ` isn't in it, so it still gets mis-tokenized)
        - The dictionary also requires a **strict majority** (>50% of annotations), not just a plurality — some source entries record their "norm" from a 3-way split or an exact tie (e.g. `โมง`, a normal word for "o'clock", was tied 7/14 between "keep" and "delete," and originally defaulted to deleting it)
        - Emoji/text-emoticons map to one of 3 tags: `[pos_emoji]`, `[neg_emoji]`, `[emoji]` (unrecognized) — a coarse sentiment signal, not per-emotion granularity
        """
    )

st.markdown("**Training & evaluation workflow** — run twice (raw text, normalized text), identically except for step 2:")
st.markdown(
    """
    1. **Split** — use Wisesight's own train / validation / test split as-is: 21,628 / 2,404 / 2,671 examples. No re-shuffling, so both runs see exactly the same examples in each split.
    2. **Build inputs** — tokenize each split's text with WangchanBERTa's tokenizer (max 64 tokens); this is the *only* step that differs between the two runs (raw `texts` vs. `normalize_thai()` output).
    3. **Train** — fine-tune for 3 epochs on the **train** split only. The model never sees validation or test examples during this step.
    4. **Validate every epoch** — after each epoch, run inference on the **validation** split and log accuracy/macro-F1. This checks training is progressing normally; no early stopping or checkpoint selection is done — it's a fixed 3-epoch run either way.
    5. **Test once, at the end** — after all 3 epochs, run one final pass on the **test** split, which the model has never seen in steps 3 or 4. This is the only number reported as "the" result — it's what the metrics below come from.
    """
)
with st.expander("Full training details"):
    st.markdown(
        """
        - New 4-class classification head on top of `airesearch/wangchanberta-base-att-spm-uncased`, randomly initialized
        - Loss: cross-entropy · Optimizer: AdamW · LR: 5e-5 with linear decay and 10% warmup
        - 3 epochs, batch size 32, max sequence length 64 tokens
        - Same random seed (42) for both runs, so weight initialization and data order match —
          the only intentional difference is the input text
        """
    )

st.header("Results & Analysis")
st.markdown(
    "All figures below come from a single held-out **test set** (2,671 examples) that neither "
    "model saw during training or validation — the same split, evaluated twice, once per variant."
)

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
        f":orange[**macro-F1 improves by +1.2pp**] — almost entirely from the minority classes: "
        "Positive +3.2pp, Question +1.9pp."
    )

    with st.expander("Confusion matrices (test set)"):
        c1, c2 = st.columns(2)
        c1.caption("Raw text")
        c1.dataframe(confusion_df(raw_metrics["confusion_matrix"]))
        c2.caption("Normalized text")
        c2.dataframe(confusion_df(norm_metrics["confusion_matrix"]))
        st.caption(
            "Reading a row shows where that true class's examples ended up. On raw text, both "
            "Positive (197 correct vs. 258 confused as Neutral) and Question (24 vs. 30) are "
            "more often mistaken for Neutral than classified correctly. Normalization flips this "
            "for Positive (226 correct vs. 220 confused) but Question stays borderline (27 vs. "
            "28) — consistent with Neutral's ~55% training share giving the model a strong prior "
            "toward it whenever a signal is ambiguous."
        )

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
        f"changing raw accuracy. Only :blue[**{len(fixed_wc)}/{len(fixed)}**] of the fixes came "
        "from an actual correction (elongation/emoji/slang) — the rest were fixed by tokenizer "
        "spacing alone, with no correction logged at all."
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

st.header("Limitations")
st.markdown(
    """
**Data and coverage**
- Dictionary coverage is incomplete — only slang seen ≥3 times in MultiLexNorm++'s corpus is
  corrected, so gaps remain (e.g. `ชิมิ` isn't in it and still gets mis-tokenized)
- Emoji mapping is coarse — all emoji collapse into 3 tags (positive/negative/neutral), not
  per-emotion granularity

**Method**
- Sarcasm and tone are out of scope — `normalize_thai()` only fixes surface form, not meaning
  that depends on context
- Part of the measured gain is a tokenizer-spacing artifact rather than a linguistic correction —
  see the fix breakdown above

**Evaluation**
- :red[Metrics are from a single training run], not averaged over multiple random seeds, so the
  precise magnitude of the +1.2pp gain — though not its direction — should be treated as an
  estimate rather than a guaranteed figure
"""
)

st.header("Conclusion")
st.markdown(
    """
Normalizing informal Thai text measurably improves :orange[**macro-F1 by +1.2pp**] on Wisesight
Sentiment, concentrated in the classes with the fewest training examples, while leaving raw
accuracy roughly unchanged — it fixes about as many predictions as it breaks, but the two sets
of changes are not distributed evenly across classes. About a third of the fixes trace back to
tokenizer spacing rather than the normalization logic itself, a distinction worth isolating in
any follow-up work.
"""
)
st.markdown(
    """
**Recommended next steps**
1. Re-run the benchmark with multiple random seeds to report a confidence interval on the macro-F1 gain, addressing the single-run limitation above
2. Separate the tokenizer-spacing effect from true corrections as its own ablation, to quantify normalize_thai()'s standalone contribution
3. Extend dictionary coverage below the ≥3-occurrence threshold using a confidence-weighted rule instead of a hard cutoff
"""
)

st.divider()

# ---------------------------------------------------------------------------
# Live demo
# ---------------------------------------------------------------------------

st.header("Live Demo")

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
