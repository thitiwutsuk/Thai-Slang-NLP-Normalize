"""Phase 1 demo: run clean_text() + tokenize_text() over real Wisesight rows."""

from datasets import load_dataset

from src.cleaning import clean_text
from src.tokenization import tokenize_text


def main() -> None:
    ds = load_dataset("pythainlp/wisesight_sentiment")["train"]

    for row in ds.select(range(5)):
        raw = row["texts"]
        cleaned = clean_text(raw)
        tokens = tokenize_text(cleaned)
        print("=" * 60)
        print("raw:     ", raw)
        print("cleaned: ", cleaned)
        print("tokens:  ", tokens)


if __name__ == "__main__":
    main()
