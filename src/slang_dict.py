"""Phase 2: slang -> standard-form dictionary lookup, from MultiLexNorm++."""

import json
from functools import lru_cache

from huggingface_hub import hf_hub_download

_REPO_ID = "hadung1802/mlnorm-resources"
_MIN_TOTAL_COUNT = 3  # drop entries seen too rarely to trust their majority vote


@lru_cache(maxsize=1)
def load_slang_dict(lang: str = "th") -> dict[str, str]:
    """original surface form -> majority-vote normalized form."""
    path = hf_hub_download(
        repo_id=_REPO_ID,
        repo_type="dataset",
        filename=f"normdict/{lang}.normdict.json",
    )
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    lookup = {}
    for original, entry in data["entries"].items():
        if entry.get("total", 0) < _MIN_TOTAL_COUNT:
            continue
        lookup[original] = entry["norm"]
    return lookup


def slang_lookup(token: str, lang: str = "th") -> str:
    return load_slang_dict(lang).get(token, token)
