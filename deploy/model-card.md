---
language: th
license: apache-2.0
base_model: airesearch/wangchanberta-base-att-spm-uncased
tags:
  - sentiment-analysis
  - thai
  - text-classification
datasets:
  - pythainlp/wisesight_sentiment
pipeline_tag: text-classification
---

# Thai Sentiment WangchanBERTa (normalized)

`WangchanBERTa` fine-tuned for 4-class sentiment classification (`pos` / `neu` / `neg` / `q`) on
[Wisesight Sentiment](https://huggingface.co/datasets/pythainlp/wisesight_sentiment), trained on
text passed through a custom Thai normalizer (`normalize_thai()`) that collapses elongation,
maps emoji to sentiment tags, and corrects slang via a dictionary mined from MultiLexNorm++.

- **Base model:** [`airesearch/wangchanberta-base-att-spm-uncased`](https://huggingface.co/airesearch/wangchanberta-base-att-spm-uncased)
- **Task:** Thai sentiment classification (positive / neutral / negative / question)
- **Test-set accuracy:** 73.9% · **macro-F1:** 64.5%

Full project, code, and the normalization pipeline this model is paired with:
https://github.com/thitiwutsuk/Thai-Slang-NLP-Normalize

## Usage

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("thitiwutsuk/thai-sentiment-wangchanberta")
model = AutoModelForSequenceClassification.from_pretrained("thitiwutsuk/thai-sentiment-wangchanberta")

inputs = tokenizer("อาหารอร่อยมาก", return_tensors="pt", truncation=True, max_length=64, padding="max_length")
with torch.no_grad():
    probs = torch.softmax(model(**inputs).logits, dim=-1)
print(probs)  # [pos, neu, neg, q]
```

Note: for best results, run input text through `normalize_thai()` from the project repo above
before tokenizing — this model was trained on normalized text, not raw text.
