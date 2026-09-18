"""Phase 2: slang -> standard-form dictionary lookup, from MultiLexNorm++."""

import json
from functools import lru_cache

from huggingface_hub import hf_hub_download

_REPO_ID = "hadung1802/mlnorm-resources"
_MIN_TOTAL_COUNT = 3  # drop entries seen too rarely to trust their majority vote
_MIN_MAJORITY_RATIO = 0.5  # "norm" must win a strict majority of annotations, not just a plurality


@lru_cache(maxsize=1)
def load_slang_dict(lang: str = "th") -> dict[str, str]:
    """original surface form -> majority-vote normalized form.

    Some MultiLexNorm++ entries record "norm" as whichever answer got the most
    votes even when that's under half of the total (e.g. a 3-way split, or an
    exact tie) -- e.g. "โมง" ("o'clock", a normal word) is recorded as norm=""
    (delete) on a 7/14 tie. Applying those would corrupt valid text, so entries
    without a strict majority are dropped rather than trusted.
    """
    path = hf_hub_download(
        repo_id=_REPO_ID,
        repo_type="dataset",
        filename=f"normdict/{lang}.normdict.json",
    )
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    lookup = {}
    for original, entry in data["entries"].items():
        total = entry.get("total", 0)
        if total < _MIN_TOTAL_COUNT:
            continue
        if entry.get("count", 0) / total <= _MIN_MAJORITY_RATIO:
            continue
        lookup[original] = entry["norm"]
    return lookup


def slang_lookup(token: str, lang: str = "th") -> str:
    return load_slang_dict(lang).get(token, token)
