"""
Phase 0: explore the structure of the two core datasets.

MultiLexNorm++ ("hadung1802/mlnorm-resources") is not a `load_dataset`-ready
repo -- it's a bag of resource files for a multi-language shared task. The
Thai slice we care about is `multilexnorm++/th/{train,dev,test}.norm`
(tab-separated `original<TAB>normalized` tokens, blank line = sentence
boundary, "_" = tweet/document boundary marker) plus `normdict/th.normdict.json`
(an aggregated slang -> standard-form dictionary).

Wisesight Sentiment loads normally via `datasets`, under the maintained
mirror `pythainlp/wisesight_sentiment` (the old bare "wisesight_sentiment"
id no longer resolves).

Usage:
    python scripts/explore_datasets.py
    python scripts/explore_datasets.py --dataset mlnorm
    python scripts/explore_datasets.py --dataset wisesight
"""

import argparse
import json
from collections import Counter

from datasets import load_dataset
from huggingface_hub import hf_hub_download

MLNORM_REPO = "hadung1802/mlnorm-resources"


def parse_norm_file(path: str):
    """Yield (original, normalized) token pairs from a .norm file, skipping
    blank lines (sentence boundaries) and "_" tweet-boundary markers."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            orig, norm = parts
            if orig == "_":
                continue
            yield orig, norm


def explore_mlnorm(lang: str = "th") -> None:
    print(f"\n{'=' * 60}\nMultiLexNorm++ [{lang}]  ({MLNORM_REPO})\n{'=' * 60}")

    for split in ("train", "dev", "test"):
        try:
            path = hf_hub_download(
                repo_id=MLNORM_REPO,
                repo_type="dataset",
                filename=f"multilexnorm++/{lang}/{split}.norm",
            )
        except Exception as exc:
            print(f"[skip] {split}: {exc}")
            continue

        pairs = list(parse_norm_file(path))
        changed = [(o, n) for o, n in pairs if o != n]
        print(f"\n--- {split}.norm: {len(pairs)} tokens, {len(changed)} changed ---")
        for orig, norm in changed[:10]:
            print(f"  {orig!r} -> {norm!r}")

    try:
        dict_path = hf_hub_download(
            repo_id=MLNORM_REPO,
            repo_type="dataset",
            filename=f"normdict/{lang}.normdict.json",
        )
        with open(dict_path, encoding="utf-8") as f:
            normdict = json.load(f)
        meta = normdict.get("_meta", {})
        entries = normdict.get("entries", {})
        tiers = Counter(e.get("tier") for e in entries.values())
        print(f"\n--- {lang}.normdict.json ---")
        print("meta:", meta)
        print("tier distribution:", dict(tiers))
        sample_keys = list(entries.keys())[:5]
        for k in sample_keys:
            print(f"  {k!r}: {entries[k]}")
    except Exception as exc:
        print(f"[skip] normdict: {exc}")


def explore_wisesight() -> None:
    print(f"\n{'=' * 60}\nWisesight Sentiment  (pythainlp/wisesight_sentiment)\n{'=' * 60}")
    ds = load_dataset("pythainlp/wisesight_sentiment")
    print(ds)

    for split in ds:
        names = ds[split].features["category"].names
        counts = Counter(ds[split]["category"])
        dist = {names[k]: v for k, v in sorted(counts.items())}
        print(f"\n--- split: {split} ({len(ds[split])} rows) --- label dist: {dist}")
        for row in ds[split].select(range(min(3, len(ds[split])))):
            print(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mlnorm", "wisesight", "all"], default="all")
    parser.add_argument("--lang", default="th", help="MultiLexNorm++ language code")
    args = parser.parse_args()

    if args.dataset in ("mlnorm", "all"):
        explore_mlnorm(args.lang)

    if args.dataset in ("wisesight", "all"):
        explore_wisesight()


if __name__ == "__main__":
    main()
