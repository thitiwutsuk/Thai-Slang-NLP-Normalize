"""Copy the raw dataset files this project depends on from the HuggingFace
cache into data/raw/, so they exist as real files inside the repo instead of
only in the hub cache.

Usage:
    python -m scripts.fetch_data
"""

import json
import shutil
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download

DATA_DIR = Path("data/raw")
MLNORM_REPO = "hadung1802/mlnorm-resources"


def fetch_mlnorm(lang: str = "th") -> None:
    out_dir = DATA_DIR / "mlnorm" / lang
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "dev", "test"):
        path = hf_hub_download(
            repo_id=MLNORM_REPO,
            repo_type="dataset",
            filename=f"multilexnorm++/{lang}/{split}.norm",
        )
        shutil.copyfile(path, out_dir / f"{split}.norm")
        print(f"wrote {out_dir / f'{split}.norm'}")

    dict_path = hf_hub_download(
        repo_id=MLNORM_REPO,
        repo_type="dataset",
        filename=f"normdict/{lang}.normdict.json",
    )
    shutil.copyfile(dict_path, DATA_DIR / "mlnorm" / f"{lang}.normdict.json")
    print(f"wrote {DATA_DIR / 'mlnorm' / f'{lang}.normdict.json'}")


def fetch_wisesight() -> None:
    out_dir = DATA_DIR / "wisesight"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("pythainlp/wisesight_sentiment")
    for split in ds:
        names = ds[split].features["category"].names
        out_path = out_dir / f"{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for row in ds[split]:
                f.write(
                    json.dumps(
                        {"text": row["texts"], "label": names[row["category"]]},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"wrote {out_path} ({len(ds[split])} rows)")


def main() -> None:
    fetch_mlnorm()
    fetch_wisesight()


if __name__ == "__main__":
    main()
