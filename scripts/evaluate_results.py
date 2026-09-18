"""Phase 4: compare raw vs. normalized results, curate fixed error cases."""

import json

LABEL_NAMES = ["pos", "neu", "neg", "q"]


def load_metrics(variant: str, results_dir: str = "results") -> dict:
    with open(f"{results_dir}/{variant}_metrics.json", encoding="utf-8") as f:
        return json.load(f)


def load_predictions(variant: str, results_dir: str = "results") -> list[dict]:
    rows = []
    with open(f"{results_dir}/{variant}_predictions.jsonl", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def format_confusion_matrix(cm: list[list[int]]) -> str:
    header = "        " + " ".join(f"{n:>5}" for n in LABEL_NAMES)
    lines = [header]
    for name, row in zip(LABEL_NAMES, cm):
        lines.append(f"true {name:<3}" + " ".join(f"{v:>5}" for v in row))
    return "\n".join(lines)


def curate_fixed_cases(raw_preds: list[dict], norm_preds: list[dict], limit: int = 10) -> list[dict]:
    fixed = []
    for raw_row, norm_row in zip(raw_preds, norm_preds):
        if not raw_row["correct"] and norm_row["correct"]:
            fixed.append(
                {
                    "raw_text": raw_row["text"],
                    "normalized_text": norm_row["text"],
                    "true_label": raw_row["true_label"],
                    "raw_pred": raw_row["pred_label"],
                    "normalized_pred": norm_row["pred_label"],
                }
            )
    return fixed[:limit]


def main() -> None:
    raw_metrics = load_metrics("raw")
    norm_metrics = load_metrics("normalized")
    raw_preds = load_predictions("raw")
    norm_preds = load_predictions("normalized")

    print("=" * 60)
    print(f"{'Setup':<20}{'Accuracy':>12}{'Macro-F1':>12}")
    print(f"{'Raw text':<20}{raw_metrics['accuracy']:>12.4f}{raw_metrics['macro_f1']:>12.4f}")
    print(f"{'Normalized text':<20}{norm_metrics['accuracy']:>12.4f}{norm_metrics['macro_f1']:>12.4f}")

    print("\nPer-class F1:")
    print(f"{'label':<8}{'raw':>10}{'normalized':>12}")
    for label in LABEL_NAMES:
        print(f"{label:<8}{raw_metrics['per_class_f1'][label]:>10.4f}{norm_metrics['per_class_f1'][label]:>12.4f}")

    print("\nConfusion matrix (raw):")
    print(format_confusion_matrix(raw_metrics["confusion_matrix"]))
    print("\nConfusion matrix (normalized):")
    print(format_confusion_matrix(norm_metrics["confusion_matrix"]))

    fixed_cases = curate_fixed_cases(raw_preds, norm_preds)
    print(f"\n{len(fixed_cases)} example cases normalization fixed (raw wrong -> normalized correct):")
    for case in fixed_cases:
        print("-" * 40)
        print("raw:       ", case["raw_text"])
        print("normalized:", case["normalized_text"])
        print(f"true={case['true_label']}  raw_pred={case['raw_pred']}  normalized_pred={case['normalized_pred']}")


if __name__ == "__main__":
    main()
