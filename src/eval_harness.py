"""
eval_harness.py - Intent Classification Evaluation Harness for AmericanAir Support Agent

Evaluates classifiers against the hand-labelled ground truth (final_intent) in
data/evaluation/golden_set.csv. Computes accuracy, macro/weighted F1, confusion
matrix, and misclassified error analysis, saving a comprehensive report to
data/evaluation/baseline_results.txt.
"""

import os
import sys
import argparse
from typing import Callable, Dict, List, Any
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

# Ensure UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure repository root is in sys.path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.baseline_classifier import DeterministicRuleClassifier, TAXONOMY

DEFAULT_INPUT_CSV = os.path.join("data", "evaluation", "golden_set.csv")
DEFAULT_OUTPUT_TXT = os.path.join("data", "evaluation", "baseline_results.txt")

def evaluate_classifier(
    classifier_fn: Callable[[str, str], str],
    input_csv: str = DEFAULT_INPUT_CSV,
    model_name: str = "Baseline 1: Deterministic Rule-Based Classifier",
) -> Dict[str, Any]:
    """
    Run evaluation for a given classifier function on the golden set.
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input evaluation file not found: {input_csv}")

    df = pd.read_csv(input_csv, keep_default_na=False)

    # Filter for labeled records
    labeled_df = df[df["final_intent"].str.strip() != ""].copy()
    if len(labeled_df) == 0:
        raise ValueError("No labeled examples found with non-empty final_intent.")

    # Generate predictions
    predictions = []
    for _, row in labeled_df.iterrows():
        pred = classifier_fn(row["customer_message"], row["conversation_context"])
        predictions.append(pred)

    labeled_df["predicted_intent"] = predictions

    y_true = labeled_df["final_intent"].tolist()
    y_pred = labeled_df["predicted_intent"].tolist()

    # Overall metrics
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Per-intent metrics
    labels = TAXONOMY
    p_per_class, r_per_class, f1_per_class, s_per_class = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    per_intent_metrics = {}
    for idx, label in enumerate(labels):
        per_intent_metrics[label] = {
            "precision": float(p_per_class[idx]),
            "recall": float(r_per_class[idx]),
            "f1": float(f1_per_class[idx]),
            "support": int(s_per_class[idx]),
        }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Misclassified examples
    misclassified = []
    for _, row in labeled_df.iterrows():
        if row["predicted_intent"] != row["final_intent"]:
            misclassified.append({
                "example_id": int(row["example_id"]),
                "conversation_id": int(row["conversation_id"]),
                "tweet_id": int(row["tweet_id"]),
                "difficulty": row["difficulty"],
                "candidate_intent": row["candidate_intent"],
                "true_intent": row["final_intent"],
                "predicted_intent": row["predicted_intent"],
                "customer_message": row["customer_message"],
                "conversation_context": row["conversation_context"],
                "notes": row["notes"],
            })

    return {
        "model_name": model_name,
        "total_examples": len(labeled_df),
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_p,
        "weighted_recall": weighted_r,
        "weighted_f1": weighted_f1,
        "per_intent_metrics": per_intent_metrics,
        "confusion_matrix": cm,
        "labels": labels,
        "misclassified": misclassified,
    }

def format_report(results: Dict[str, Any]) -> str:
    """Format evaluation results into a human-readable text report."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"AMERICAN AIRLINES INTENT CLASSIFICATION EVALUATION REPORT")
    lines.append(f"Model: {results['model_name']}")
    lines.append("=" * 80)
    lines.append(f"Total Evaluated Examples: {results['total_examples']}")
    lines.append(f"Accuracy:                {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    lines.append(f"Macro Precision:         {results['macro_precision']:.4f}")
    lines.append(f"Macro Recall:            {results['macro_recall']:.4f}")
    lines.append(f"Macro F1-Score:          {results['macro_f1']:.4f}")
    lines.append(f"Weighted F1-Score:       {results['weighted_f1']:.4f}")
    lines.append("")

    # Per-intent metrics table
    lines.append("-" * 80)
    lines.append("PER-INTENT PERFORMANCE BREAKDOWN")
    lines.append("-" * 80)
    header = f"{'Intent':<30} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}"
    lines.append(header)
    lines.append("-" * len(header))
    for intent, metrics in results["per_intent_metrics"].items():
        lines.append(
            f"{intent:<30} | {metrics['precision']:<10.4f} | {metrics['recall']:<10.4f} | "
            f"{metrics['f1']:<10.4f} | {metrics['support']:<8d}"
        )
    lines.append("-" * len(header))
    lines.append("")

    # Confusion matrix
    lines.append("-" * 80)
    lines.append("CONFUSION MATRIX")
    lines.append("Rows = Ground Truth (final_intent), Columns = Predicted Intent")
    lines.append("-" * 80)
    # Abbreviated header for compactness
    abbr_labels = [f"[{i+1}]" for i in range(len(results["labels"]))]
    lines.append("Legend:")
    for i, label in enumerate(results["labels"]):
        lines.append(f"  [{i+1:<2}] {label}")
    lines.append("")

    cm_header = f"{'Actual':<33} " + " ".join([f"{abbr:>5}" for abbr in abbr_labels]) + f" {'Total':>7}"
    lines.append(cm_header)
    lines.append("-" * len(cm_header))

    cm = results["confusion_matrix"]
    for i, label in enumerate(results["labels"]):
        row_str = f"[{i+1:<2}] {label:<27} "
        row_vals = " ".join([f"{val:>5d}" for val in cm[i]])
        row_total = sum(cm[i])
        lines.append(f"{row_str}{row_vals} {row_total:>7d}")
    lines.append("-" * len(cm_header))
    lines.append("")

    # Misclassified examples breakdown
    misclassified = results["misclassified"]
    lines.append("=" * 80)
    lines.append(f"MISCLASSIFIED EXAMPLES ANALYSIS (Total: {len(misclassified)} / {results['total_examples']})")
    lines.append("=" * 80)

    # Breakdown by difficulty
    diff_counts = {}
    for item in misclassified:
        d = item["difficulty"]
        diff_counts[d] = diff_counts.get(d, 0) + 1
    lines.append(f"Error Distribution by Difficulty: " + ", ".join([f"{k}: {v}" for k, v in diff_counts.items()]))
    lines.append("")

    for idx, item in enumerate(misclassified, 1):
        lines.append(f"Error #{idx:02d} [Example #{item['example_id']}] (Difficulty: {item['difficulty']})")
        lines.append(f"  - Ground Truth Intent : {item['true_intent']}")
        lines.append(f"  - Predicted Intent    : {item['predicted_intent']}")
        lines.append(f"  - Candidate Heuristic : {item['candidate_intent']}")
        if item["conversation_context"] and "[No preceding" not in item["conversation_context"]:
            ctx_summary = item["conversation_context"].replace("\n", " ")
            if len(ctx_summary) > 120:
                ctx_summary = ctx_summary[:120] + "..."
            lines.append(f"  - Preceding Context   : {ctx_summary}")
        clean_msg = item["customer_message"].replace("\n", " ")
        lines.append(f"  - Customer Message    : \"{clean_msg}\"")
        if item["notes"]:
            lines.append(f"  - Sampling Notes      : {item['notes']}")
        lines.append("-" * 60)

    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Evaluation Harness for AmericanAir Intent Classification")
    parser.add_argument("--input", default=DEFAULT_INPUT_CSV, help="Path to golden set CSV (default: data/evaluation/golden_set.csv)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_TXT, help="Path to save evaluation report (default: data/evaluation/baseline_results.txt)")
    parser.add_argument("--quiet", action="store_true", help="Do not print full report to stdout")
    args = parser.parse_args()

    # Initialize Baseline 1
    classifier = DeterministicRuleClassifier()

    # Run evaluation
    results = evaluate_classifier(
        classifier_fn=classifier.predict,
        input_csv=args.input,
        model_name="Baseline 1: Deterministic Rule-Based Classifier",
    )

    # Format report
    report_text = format_report(results)

    # Save report to file
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report_text)

    # Print summary or full report
    if not args.quiet:
        print(report_text)
    else:
        print(f"Evaluation complete. Accuracy: {results['accuracy']:.4f}, Macro F1: {results['macro_f1']:.4f}")

    print(f"\nSaved human-readable evaluation report to: {args.output}")

if __name__ == "__main__":
    main()
