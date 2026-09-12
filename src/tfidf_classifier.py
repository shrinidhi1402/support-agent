"""
tfidf_classifier.py - Baseline 2: TF-IDF + Logistic Regression Intent Classifier

Trains a TF-IDF + Logistic Regression classifier on 47,244 historical customer
messages weakly labeled by Baseline 1 (DeterministicRuleClassifier).
Evaluates strictly on the held-out 200-example golden set (golden_set.csv).
All 200 golden-set tweet IDs are strictly excluded from training.
"""

import os
import sys
import argparse
import time
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
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

# Ensure project root is in sys.path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.baseline_classifier import DeterministicRuleClassifier, TAXONOMY

CONV_FILE = os.path.join("data", "processed", "americanair_conversations.csv")
GOLDEN_FILE = os.path.join("data", "evaluation", "golden_set.csv")
OUTPUT_REPORT_FILE = os.path.join("data", "evaluation", "tfidf_results.txt")

def build_context_map(df_conv: pd.DataFrame) -> Dict[Tuple[int, int], str]:
    """
    Build a fast mapping from (conversation_id, position) -> preceding context string.
    Runs linearly in O(N).
    """
    context_map = {}
    curr_conv = None
    history = []

    # Iterate linearly over sorted conversation turns
    for cid, pos, author, text in zip(
        df_conv["conversation_id"],
        df_conv["position"],
        df_conv["author_id"],
        df_conv["text"],
    ):
        cid = int(cid)
        pos = int(pos)
        if cid != curr_conv:
            curr_conv = cid
            history = []

        author_str = "AmericanAir" if author == "AmericanAir" else f"Customer ({author})"
        clean_text = str(text).replace("\n", " ").strip()
        context_map[(cid, pos)] = " | ".join(history) if history else ""
        history.append(f"[{pos}] {author_str}: \"{clean_text}\"")

    return context_map

class TfidfIntentClassifier:
    """
    Baseline 2 Intent Classifier combining TF-IDF N-gram feature extraction
    with class-weighted Logistic Regression.
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        max_features: int = 40000,
        min_df: int = 2,
        sublinear_tf: bool = True,
        C: float = 1.0,
        class_weight: str = "balanced",
        random_state: int = 42,
    ):
        self.pipeline = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=ngram_range,
                    max_features=max_features,
                    min_df=min_df,
                    sublinear_tf=sublinear_tf,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    class_weight=class_weight,
                    max_iter=1000,
                    random_state=random_state,
                    solver="lbfgs",
                ),
            ),
        ])

    def fit(self, X: List[str], y: List[str]):
        """Train pipeline on weakly labeled examples."""
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: List[str]) -> List[str]:
        """Predict intent classes."""
        return self.pipeline.predict(X).tolist()

    def predict_proba(self, X: List[str]) -> np.ndarray:
        """Predict probability distributions."""
        return self.pipeline.predict_proba(X)

def prepare_training_data(
    conv_path: str = CONV_FILE,
    golden_path: str = GOLDEN_FILE,
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """
    Load historical customer messages, exclude golden-set tweet IDs,
    and generate weak labels using Baseline 1.
    """
    print(f"Loading golden set from: {golden_path}")
    golden_df = pd.read_csv(golden_path, keep_default_na=False)
    golden_tweet_ids = set(golden_df["tweet_id"].astype(int))
    print(f"  Golden set size: {len(golden_df)} (Tweet IDs to exclude: {len(golden_tweet_ids)})")

    print(f"Loading historical conversations from: {conv_path}")
    t0 = time.time()
    df_conv = pd.read_csv(conv_path, low_memory=False)
    df_conv = df_conv.sort_values(["conversation_id", "position"])
    print(f"  Loaded {len(df_conv):,} messages in {time.time() - t0:.2f}s.")

    print("Building conversation context mapping...")
    t1 = time.time()
    context_map = build_context_map(df_conv)
    print(f"  Context mapping completed in {time.time() - t1:.2f}s.")

    # Filter customer messages
    cust_df = df_conv[df_conv["inbound"] == True].copy()
    total_cust_msgs = len(cust_df)

    # Exclude all golden-set tweet IDs
    train_df = cust_df[~cust_df["tweet_id"].astype(int).isin(golden_tweet_ids)].copy()
    num_train = len(train_df)
    print(f"  Total customer messages: {total_cust_msgs:,}")
    print(f"  Training examples after strict golden-set exclusion: {num_train:,}")
    assert total_cust_msgs - num_train == len(golden_df), (
        f"Mismatch in golden set exclusion count: {total_cust_msgs - num_train} vs {len(golden_df)}"
    )

    # Generate weak labels using Baseline 1
    print("Generating weak labels using Baseline 1 (DeterministicRuleClassifier)...")
    rule_clf = DeterministicRuleClassifier()
    train_texts = []
    weak_labels = []

    t2 = time.time()
    for cid, pos, msg in zip(
        train_df["conversation_id"], train_df["position"], train_df["text"]
    ):
        msg_str = str(msg).strip()
        ctx = context_map.get((int(cid), int(pos)), "")
        label = rule_clf.predict(msg_str, ctx)

        # Construct input: preceding context + current message
        if ctx:
            full_input = f"{ctx} \n {msg_str}"
        else:
            full_input = msg_str

        train_texts.append(full_input)
        weak_labels.append(label)

    print(f"  Weak labeling finished in {time.time() - t2:.2f}s.")

    # Compute distribution
    unique_labels, counts = np.unique(weak_labels, return_counts=True)
    label_dist = {str(k): int(v) for k, v in zip(unique_labels, counts)}
    # Sort descending by count
    label_dist = dict(sorted(label_dist.items(), key=lambda x: x[1], reverse=True))

    meta = {
        "num_train": num_train,
        "label_dist": label_dist,
        "golden_df": golden_df,
    }
    return train_texts, weak_labels, meta

def evaluate_model(
    model: TfidfIntentClassifier,
    golden_df: pd.DataFrame,
    weak_label_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate trained TF-IDF model on golden set.
    """
    # Construct evaluation inputs
    golden_inputs = []
    for _, row in golden_df.iterrows():
        msg = str(row["customer_message"]).strip()
        ctx = str(row["conversation_context"]).strip()
        if ctx and "[No preceding" not in ctx:
            full_input = f"{ctx} \n {msg}"
        else:
            full_input = msg
        golden_inputs.append(full_input)

    golden_y = golden_df["final_intent"].tolist()

    # Predict
    y_pred = model.predict(golden_inputs)

    # Overall metrics
    acc = accuracy_score(golden_y, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        golden_y, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        golden_y, y_pred, average="weighted", zero_division=0
    )

    # Per-intent metrics
    labels = TAXONOMY
    p_per_class, r_per_class, f1_per_class, s_per_class = precision_recall_fscore_support(
        golden_y, y_pred, labels=labels, zero_division=0
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
    cm = confusion_matrix(golden_y, y_pred, labels=labels)

    # Misclassified examples
    misclassified = []
    for idx, row in golden_df.iterrows():
        pred = y_pred[idx]
        true_val = row["final_intent"]
        if pred != true_val:
            misclassified.append({
                "example_id": int(row["example_id"]),
                "conversation_id": int(row["conversation_id"]),
                "tweet_id": int(row["tweet_id"]),
                "difficulty": row["difficulty"],
                "candidate_intent": row["candidate_intent"],
                "true_intent": true_val,
                "predicted_intent": pred,
                "customer_message": row["customer_message"],
                "conversation_context": row["conversation_context"],
                "notes": row["notes"],
            })

    return {
        "model_name": "Baseline 2: TF-IDF + Logistic Regression (Weak Supervision)",
        "num_train": weak_label_meta["num_train"],
        "weak_label_dist": weak_label_meta["label_dist"],
        "total_evaluated": len(golden_df),
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
    """Format TF-IDF evaluation results into a comprehensive text report."""
    lines = []
    lines.append("=" * 80)
    lines.append("AMERICAN AIRLINES INTENT CLASSIFICATION EVALUATION REPORT")
    lines.append(f"Model: {results['model_name']}")
    lines.append("=" * 80)
    lines.append(f"Training Dataset:        47,244 historical customer messages (weakly labeled)")
    lines.append(f"Evaluation Dataset:      200 held-out golden set examples (hand-labeled ground truth)")
    lines.append("")

    # Weak label distribution
    lines.append("-" * 80)
    lines.append("WEAK-LABEL TRAINING DISTRIBUTION (47,244 Examples)")
    lines.append("-" * 80)
    total_train = results["num_train"]
    for intent, count in results["weak_label_dist"].items():
        pct = (count / total_train) * 100
        lines.append(f"  {intent:<30}: {count:>6d} ({pct:>5.1f}%)")
    lines.append("")

    # Overall metrics
    lines.append("-" * 80)
    lines.append("GOLDEN SET OVERALL METRICS (Held-Out Evaluation)")
    lines.append("-" * 80)
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

    # Misclassified examples
    misclassified = results["misclassified"]
    lines.append("=" * 80)
    lines.append(f"MISCLASSIFIED GOLDEN EXAMPLES ANALYSIS (Total: {len(misclassified)} / {results['total_evaluated']})")
    lines.append("=" * 80)

    diff_counts = {}
    for item in misclassified:
        d = item["difficulty"]
        diff_counts[d] = diff_counts.get(d, 0) + 1
    lines.append("Error Distribution by Difficulty: " + ", ".join([f"{k}: {v}" for k, v in diff_counts.items()]))
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
    parser = argparse.ArgumentParser(description="Baseline 2: TF-IDF + Logistic Regression Intent Classifier")
    parser.add_argument("--conv-file", default=CONV_FILE, help="Path to reconstructed conversations CSV")
    parser.add_argument("--golden-file", default=GOLDEN_FILE, help="Path to golden set evaluation CSV")
    parser.add_argument("--output", default=OUTPUT_REPORT_FILE, help="Path to save evaluation report")
    args = parser.parse_args()

    # Step 1: Prepare weakly labeled training data
    train_texts, weak_labels, meta = prepare_training_data(
        conv_path=args.conv_file,
        golden_path=args.golden_file,
    )

    # Step 2: Initialize & train classifier
    print("\nInitializing TfidfIntentClassifier (TF-IDF + Logistic Regression)...")
    clf = TfidfIntentClassifier(
        ngram_range=(1, 2),
        max_features=40000,
        min_df=2,
        sublinear_tf=True,
        C=1.0,
        class_weight="balanced",
        random_state=42,
    )

    print("Fitting model on 47,244 training examples...")
    t0 = time.time()
    clf.fit(train_texts, weak_labels)
    print(f"  Model trained in {time.time() - t0:.2f}s.")

    # Step 3: Evaluate on held-out golden set
    print("Evaluating model strictly on held-out golden set (200 examples)...")
    results = evaluate_model(
        model=clf,
        golden_df=meta["golden_df"],
        weak_label_meta=meta,
    )

    # Step 4: Format & write report
    report_text = format_report(results)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report_text)

    # Print report
    print("\n" + report_text)
    print(f"\nSuccessfully saved full evaluation report to: {args.output}")

if __name__ == "__main__":
    main()
