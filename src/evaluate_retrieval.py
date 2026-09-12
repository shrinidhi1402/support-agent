"""
evaluate_retrieval.py - Evaluation Harness for Historical Interaction Retrieval (Baseline 3)

Evaluates HistoricalInteractionRetriever against all 200 golden-set examples in
data/evaluation/golden_set.csv. Computes Top-1/Top-3 cosine similarity and
intent alignment, analyzes representative successes and failures, and saves a
human-readable report to data/evaluation/retrieval_results.txt.
"""

import os
import sys
import argparse
import time
from typing import Dict, List, Any
import pandas as pd
import numpy as np

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

from src.retrieval import HistoricalInteractionRetriever, CONV_FILE, GOLDEN_FILE
from src.baseline_classifier import TAXONOMY

DEFAULT_OUTPUT_TXT = os.path.join("data", "evaluation", "retrieval_results.txt")

def evaluate_retrieval(
    retriever: HistoricalInteractionRetriever,
    golden_file: str = GOLDEN_FILE,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Evaluate retriever against all hand-labeled examples in the golden set.
    """
    golden_df = pd.read_csv(golden_file, keep_default_na=False)

    top1_sims = []
    top3_sims = []
    top1_intent_matches = []
    top3_intent_matches = []

    per_intent_stats = {
        intent: {"count": 0, "top1_sims": [], "top3_sims": [], "top1_matches": 0, "top3_matches": 0}
        for intent in TAXONOMY
    }

    eval_records = []

    t0 = time.time()
    for idx, row in golden_df.iterrows():
        example_id = int(row["example_id"])
        golden_intent = str(row["final_intent"]).strip()
        customer_msg = str(row["customer_message"]).strip()
        context_str = str(row["conversation_context"]).strip()
        difficulty = str(row["difficulty"]).strip()
        notes = str(row["notes"]).strip()

        # Retrieve top_k results
        results = retriever.retrieve(customer_msg, context_str, top_k=top_k)

        top1_res = results[0] if results else None
        top1_sim = top1_res["similarity_score"] if top1_res else 0.0
        top1_match = (top1_res["historical_intent"] == golden_intent) if top1_res else False

        top3_scores = [r["similarity_score"] for r in results]
        top3_sim = float(np.mean(top3_scores)) if top3_scores else 0.0
        top3_intents = [r["historical_intent"] for r in results]
        top3_match = (golden_intent in top3_intents)

        top1_sims.append(top1_sim)
        top3_sims.append(top3_sim)
        top1_intent_matches.append(top1_match)
        top3_intent_matches.append(top3_match)

        # Record per-intent stats
        if golden_intent in per_intent_stats:
            per_intent_stats[golden_intent]["count"] += 1
            per_intent_stats[golden_intent]["top1_sims"].append(top1_sim)
            per_intent_stats[golden_intent]["top3_sims"].append(top3_sim)
            if top1_match:
                per_intent_stats[golden_intent]["top1_matches"] += 1
            if top3_match:
                per_intent_stats[golden_intent]["top3_matches"] += 1

        eval_records.append({
            "example_id": example_id,
            "difficulty": difficulty,
            "golden_intent": golden_intent,
            "customer_message": customer_msg,
            "context": context_str,
            "notes": notes,
            "results": results,
            "top1_similarity": top1_sim,
            "top3_similarity": top3_sim,
            "top1_match": top1_match,
            "top3_match": top3_match,
        })

    eval_duration = time.time() - t0

    # Sort for best and worst examples
    # Successful: Top-3 matches intent, sorted by highest top-1 similarity
    successful_cases = sorted(
        [r for r in eval_records if r["top1_match"]],
        key=lambda x: x["top1_similarity"],
        reverse=True,
    )

    # Poor: Intent mismatch or low similarity
    poor_cases = sorted(
        [r for r in eval_records if not r["top3_match"]],
        key=lambda x: x["top1_similarity"],
    )

    return {
        "total_evaluated": len(golden_df),
        "corpus_size": len(retriever.corpus),
        "excluded_golden_convs": len(retriever.golden_conv_ids),
        "eval_duration_sec": eval_duration,
        "mean_top1_similarity": float(np.mean(top1_sims)),
        "mean_top3_similarity": float(np.mean(top3_sims)),
        "top1_intent_accuracy": float(np.mean(top1_intent_matches)),
        "top1_intent_count": int(sum(top1_intent_matches)),
        "top3_intent_accuracy": float(np.mean(top3_intent_matches)),
        "top3_intent_count": int(sum(top3_intent_matches)),
        "per_intent_stats": per_intent_stats,
        "successful_cases": successful_cases,
        "poor_cases": poor_cases,
        "all_records": eval_records,
    }

def format_report(results: Dict[str, Any]) -> str:
    """Format retrieval evaluation results into a comprehensive text report."""
    lines = []
    lines.append("=" * 80)
    lines.append("AMERICAN AIRLINES HISTORICAL RETRIEVAL EVALUATION REPORT")
    lines.append("Baseline 3: TF-IDF Unigrams + Bigrams Cosine Similarity")
    lines.append("=" * 80)
    lines.append(f"Retrieval Corpus Size:         {results['corpus_size']:,} interactions (customer -> agent)")
    lines.append(f"Golden Set Conversations Excluded: {results['excluded_golden_convs']} (0 leakage)")
    lines.append(f"Evaluated Golden Examples:     {results['total_evaluated']}")
    lines.append(f"Evaluation Time:               {results['eval_duration_sec']:.2f}s")
    lines.append("")

    lines.append("-" * 80)
    lines.append("HEADLINE RETRIEVAL METRICS (Top-k = 3)")
    lines.append("-" * 80)
    lines.append(f"1. Number Evaluated:                  {results['total_evaluated']}")
    lines.append(f"2. Mean Top-1 Cosine Similarity:       {results['mean_top1_similarity']:.4f}")
    lines.append(f"3. Mean Top-3 Cosine Similarity:       {results['mean_top3_similarity']:.4f}")
    lines.append(f"4. Top-1 Intent Alignment Accuracy:    {results['top1_intent_accuracy']*100:.2f}% ({results['top1_intent_count']}/{results['total_evaluated']})")
    lines.append(f"5. Top-3 Intent Alignment (% >= 1):   {results['top3_intent_accuracy']*100:.2f}% ({results['top3_intent_count']}/{results['total_evaluated']})")
    lines.append("")

    # Per-intent breakdown
    lines.append("-" * 80)
    lines.append("PER-INTENT RETRIEVAL BREAKDOWN")
    lines.append("-" * 80)
    header = f"{'Intent':<30} | {'Support':<7} | {'Top-1 Sim':<10} | {'Top-3 Sim':<10} | {'Top-1 Match':<12} | {'Top-3 Match':<12}"
    lines.append(header)
    lines.append("-" * len(header))

    for intent in TAXONOMY:
        st = results["per_intent_stats"][intent]
        cnt = st["count"]
        if cnt > 0:
            top1_s = np.mean(st["top1_sims"])
            top3_s = np.mean(st["top3_sims"])
            top1_pct = (st["top1_matches"] / cnt) * 100
            top3_pct = (st["top3_matches"] / cnt) * 100
            lines.append(
                f"{intent:<30} | {cnt:<7d} | {top1_s:<10.4f} | {top3_s:<10.4f} | "
                f"{top1_pct:>5.1f}% ({st['top1_matches']:>2d}) | {top3_pct:>5.1f}% ({st['top3_matches']:>2d})"
            )
    lines.append("-" * len(header))
    lines.append("")

    # Section: Successful Retrieval Examples
    lines.append("=" * 80)
    lines.append("REPRESENTATIVE SUCCESSFUL RETRIEVAL EXAMPLES (High Similarity & Intent Match)")
    lines.append("=" * 80)

    for i, ex in enumerate(results["successful_cases"][:5], 1):
        lines.append(f"Success #{i} [Example #{ex['example_id']}] (Difficulty: {ex['difficulty']})")
        lines.append(f"  Target Golden Intent : {ex['golden_intent']}")
        if ex["context"] and "[No preceding" not in ex["context"]:
            lines.append(f"  Preceding Context    : \"{ex['context'][:120]}...\"")
        lines.append(f"  Customer Message     : \"{ex['customer_message']}\"")
        lines.append(f"  Top-1 Similarity     : {ex['top1_similarity']:.4f}")
        lines.append("  Retrieved Historical Interaction #1:")
        top1 = ex["results"][0]
        lines.append(f"    - Historical Conv ID  : {top1['conversation_id']}")
        lines.append(f"    - Historical Intent   : {top1['historical_intent']}")
        lines.append(f"    - Historical Customer : \"{top1['customer_message']}\"")
        lines.append(f"    - AmericanAir Response: \"{top1['agent_response']}\"")
        lines.append("-" * 70)

    lines.append("")

    # Section: Poor Retrieval Examples
    lines.append("=" * 80)
    lines.append("REPRESENTATIVE POOR RETRIEVAL EXAMPLES (Intent Mismatch / Lexical Divergence)")
    lines.append("=" * 80)

    for i, ex in enumerate(results["poor_cases"][:5], 1):
        lines.append(f"Failure #{i} [Example #{ex['example_id']}] (Difficulty: {ex['difficulty']})")
        lines.append(f"  Target Golden Intent : {ex['golden_intent']}")
        if ex["context"] and "[No preceding" not in ex["context"]:
            lines.append(f"  Preceding Context    : \"{ex['context'][:120]}...\"")
        lines.append(f"  Customer Message     : \"{ex['customer_message']}\"")
        lines.append(f"  Top-1 Similarity     : {ex['top1_similarity']:.4f}")
        lines.append(f"  Sampling Notes       : {ex['notes']}")
        lines.append("  Retrieved Historical Interaction #1 (Mismatched):")
        top1 = ex["results"][0]
        lines.append(f"    - Historical Conv ID  : {top1['conversation_id']}")
        lines.append(f"    - Historical Intent   : {top1['historical_intent']} (Mismatch)")
        lines.append(f"    - Historical Customer : \"{top1['customer_message']}\"")
        lines.append(f"    - AmericanAir Response: \"{top1['agent_response']}\"")
        lines.append("-" * 70)

    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Baseline 3 Historical Interaction Retrieval")
    parser.add_argument("--golden-file", default=GOLDEN_FILE, help="Path to golden set evaluation CSV")
    parser.add_argument("--conv-file", default=CONV_FILE, help="Path to conversations CSV")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_TXT, help="Path to save evaluation report")
    parser.add_argument("--top-k", type=int, default=3, help="Number of retrieved interactions to evaluate")
    args = parser.parse_args()

    print(f"Loading HistoricalInteractionRetriever from {args.conv_file}...")
    t0 = time.time()
    retriever = HistoricalInteractionRetriever(
        conv_file=args.conv_file,
        golden_file=args.golden_file,
        top_k=args.top_k,
    )
    print(f"Retriever ready in {time.time() - t0:.2f}s. Corpus size: {len(retriever.corpus):,} interactions.")

    print(f"\nEvaluating against {args.golden_file}...")
    results = evaluate_retrieval(retriever, golden_file=args.golden_file, top_k=args.top_k)

    report_text = format_report(results)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report_text)

    print("\n" + report_text)
    print(f"\nSaved historical retrieval evaluation report to: {args.output}")

if __name__ == "__main__":
    main()
