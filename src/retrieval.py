"""
retrieval.py - Baseline 3: Historical Interaction Retrieval for AmericanAir Support Agent

Retrieves top-k similar historical customer -> AmericanAir interactions
grounded in data/processed/americanair_conversations.csv using TF-IDF word
unigrams + bigrams and cosine similarity.

All 200 golden-set examples and their complete corresponding conversations are
strictly excluded from the retrieval corpus to prevent data leakage.
"""

import os
import sys
import time
import re
from typing import Dict, List, Any, Optional, Set, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

from src.baseline_classifier import DeterministicRuleClassifier, clean_text, TAXONOMY

CONV_FILE = os.path.join("data", "processed", "americanair_conversations.csv")
GOLDEN_FILE = os.path.join("data", "evaluation", "golden_set.csv")

class HistoricalInteractionRetriever:
    """
    TF-IDF based retriever over historical AmericanAir customer-agent interactions.
    """

    def __init__(
        self,
        conv_file: str = CONV_FILE,
        golden_file: str = GOLDEN_FILE,
        ngram_range: Tuple[int, int] = (1, 2),
        max_features: int = 50000,
        min_df: int = 2,
        sublinear_tf: bool = True,
        top_k: int = 3,
    ):
        self.conv_file = conv_file
        self.golden_file = golden_file
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df = min_df
        self.sublinear_tf = sublinear_tf
        self.default_top_k = top_k

        self.rule_clf = DeterministicRuleClassifier()
        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
            stop_words="english",
        )

        self.corpus: List[Dict[str, Any]] = []
        self.X_corpus = None
        self.golden_conv_ids: Set[int] = set()
        self.golden_tweet_ids: Set[int] = set()

        self._build_index()

    def _build_index(self):
        """Load conversations, exclude golden set, and index interactions."""
        # 1. Load golden set IDs to exclude
        if os.path.exists(self.golden_file):
            golden_df = pd.read_csv(self.golden_file, keep_default_na=False)
            self.golden_conv_ids = set(golden_df["conversation_id"].astype(int))
            self.golden_tweet_ids = set(golden_df["tweet_id"].astype(int))
        else:
            self.golden_conv_ids = set()
            self.golden_tweet_ids = set()

        # 2. Load conversations
        df_conv = pd.read_csv(self.conv_file, low_memory=False)
        df_conv = df_conv.sort_values(["conversation_id", "position"])

        # 3. Extract customer -> agent response pairs linearly
        corpus = []
        curr_conv = None
        conv_turns = []

        for cid, pos, author, text, inbound, tid in zip(
            df_conv["conversation_id"],
            df_conv["position"],
            df_conv["author_id"],
            df_conv["text"],
            df_conv["inbound"],
            df_conv["tweet_id"],
        ):
            cid = int(cid)
            pos = int(pos)
            tid = int(tid)
            inbound = bool(inbound)

            # Strict exclusion of entire conversation if it contains any golden example
            if cid in self.golden_conv_ids:
                continue

            row_data = {
                "conversation_id": cid,
                "position": pos,
                "author_id": author,
                "text": text,
                "inbound": inbound,
                "tweet_id": tid,
            }

            if cid != curr_conv:
                if conv_turns:
                    self._extract_pairs_from_conv(conv_turns, corpus)
                curr_conv = cid
                conv_turns = [row_data]
            else:
                conv_turns.append(row_data)

        # Process the final conversation
        if conv_turns and curr_conv not in self.golden_conv_ids:
            self._extract_pairs_from_conv(conv_turns, corpus)

        self.corpus = corpus

        # 4. Build TF-IDF matrix
        corpus_texts = [item["clean_indexed"] for item in self.corpus]
        self.X_corpus = self.vectorizer.fit_transform(corpus_texts)

    def _extract_pairs_from_conv(
        self, conv_turns: List[Dict[str, Any]], corpus: List[Dict[str, Any]]
    ):
        """Extract adjacent customer -> AmericanAir pairs within a conversation."""
        history = []
        for idx in range(len(conv_turns)):
            turn = conv_turns[idx]
            pos = turn["position"]
            author = "AmericanAir" if turn["author_id"] == "AmericanAir" else f"Customer ({turn['author_id']})"
            text = str(turn["text"]).replace("\n", " ").strip()

            # Pair condition: inbound customer message followed by AmericanAir response
            if turn["inbound"] and idx + 1 < len(conv_turns) and not conv_turns[idx + 1]["inbound"]:
                aa_turn = conv_turns[idx + 1]
                ctx_str = " | ".join(history) if history else ""
                cust_msg = str(turn["text"]).strip()
                aa_resp = str(aa_turn["text"]).strip()

                # Clean text for vectorization
                clean_msg = clean_text(cust_msg)
                clean_ctx = clean_text(ctx_str)
                clean_indexed = f"{clean_ctx} {clean_msg}" if clean_ctx else clean_msg

                # Weak label for historical interaction intent
                hist_intent = self.rule_clf.predict(cust_msg, ctx_str)

                corpus.append({
                    "conversation_id": turn["conversation_id"],
                    "customer_position": pos,
                    "customer_tweet_id": turn["tweet_id"],
                    "agent_tweet_id": aa_turn["tweet_id"],
                    "context": ctx_str,
                    "customer_message": cust_msg,
                    "agent_response": aa_resp,
                    "clean_indexed": clean_indexed,
                    "historical_intent": hist_intent,
                })

            history.append(f"[{pos}] {author}: \"{text}\"")

    def format_query(self, customer_message: str, conversation_context: str = "") -> str:
        """Construct normalized query text from message and preceding context."""
        clean_msg = clean_text(customer_message)
        if conversation_context and "[No preceding" not in conversation_context:
            clean_ctx = clean_text(conversation_context)
            return f"{clean_ctx} {clean_msg}" if clean_ctx else clean_msg
        return clean_msg

    def retrieve(
        self,
        customer_message: str,
        conversation_context: str = "",
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most similar historical interactions for a customer message.
        Excludes near-duplicate results from the same conversation.
        """
        if top_k is None:
            top_k = self.default_top_k

        query_text = self.format_query(customer_message, conversation_context)
        q_vec = self.vectorizer.transform([query_text])

        # Cosine similarity against corpus
        sims = cosine_similarity(q_vec, self.X_corpus)[0]

        # Top indices descending
        sorted_indices = np.argsort(-sims)

        top_results = []
        seen_convs = set()

        for idx in sorted_indices:
            item = self.corpus[idx]
            cid = item["conversation_id"]

            # Avoid returning multiple near-duplicate results from the exact same conversation
            if cid in seen_convs:
                continue
            seen_convs.add(cid)

            top_results.append({
                "similarity_score": float(sims[idx]),
                "conversation_id": item["conversation_id"],
                "customer_tweet_id": item["customer_tweet_id"],
                "agent_tweet_id": item["agent_tweet_id"],
                "customer_message": item["customer_message"],
                "agent_response": item["agent_response"],
                "context": item["context"],
                "historical_intent": item["historical_intent"],
            })

            if len(top_results) == top_k:
                break

        return top_results

def main():
    print("Initializing HistoricalInteractionRetriever...")
    t0 = time.time()
    retriever = HistoricalInteractionRetriever()
    print(f"Retriever initialized in {time.time() - t0:.2f}s.")
    print(f"Corpus size: {len(retriever.corpus):,} interactions.")
    print(f"Excluded golden conversations: {len(retriever.golden_conv_ids)}")

    # Sample query
    test_msg = "My bag was lost on flight AA1234 from DFW to ORD. Where can I track it?"
    print(f"\nSample Query: \"{test_msg}\"")
    results = retriever.retrieve(test_msg, top_k=3)
    for i, res in enumerate(results, 1):
        print(f"\n--- Result #{i} (Score: {res['similarity_score']:.4f}) ---")
        print(f"Conv ID: {res['conversation_id']} | Intent: {res['historical_intent']}")
        print(f"Customer: \"{res['customer_message']}\"")
        print(f"AmericanAir: \"{res['agent_response']}\"")

if __name__ == "__main__":
    main()
