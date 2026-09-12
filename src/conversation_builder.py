"""
conversation_builder.py
=======================
Reconstructs individual customer-support conversations for AmericanAir
from the raw Twitter Customer Support (twcs.csv) dataset.

Design & Algorithm Overview
---------------------------
A true customer-support conversation represents an interaction between
ONE customer (author_id) and AmericanAir (the support agent).

In Twitter's reply structure, multiple unrelated customers often reply
to the same brand tweet (e.g. a broadcast announcement, a promotion, or
a high-visibility customer issue). A naive whole-tree BFS lumps all of
these unrelated customers into one giant conversation.

To solve this and reconstruct clean, customer-isolated conversations:
1. Load only the 7 required columns from twcs.csv.
2. Build the reply graph (child -> parent via in_response_to_tweet_id,
   augmented with response_tweet_id).
3. Pre-parse timestamps into numeric UTC epoch seconds for deterministic
   chronological sorting (with tweet_id as a monotonic tiebreaker).
4. Identify all connected reply trees containing AmericanAir interactions.
5. Decompose each tree into customer-isolated conversations:
   - If a tree contains only 1 customer: all tweets form that customer's
     conversation with AmericanAir.
   - If a tree contains multiple customers (e.g. replies to an AA broadcast
     or multiple bystanders): partition the tree by customer author_id.
     For each customer, extract their messages and all AmericanAir replies
     in their direct reply branch. Unrelated customers become separate
     conversations.
   - Restrict conversation participants strictly to the specific customer
     and AmericanAir (filtering out any third-party brand replies).
6. Enforce strict quality filters:
   - Each conversation must have >= 1 customer message and >= 1 AmericanAir response.
   - Exact 1-based position numbering per conversation.
   - Zero duplicate tweet IDs across all conversations.
7. Save output to data/processed/americanair_conversations.csv.
8. Output comprehensive validation statistics.
"""

import os
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timezone

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration & Paths
# ---------------------------------------------------------------------------
INPUT_FILE  = os.path.join("data", "raw", "twcs.csv")
OUTPUT_DIR  = os.path.join("data", "processed")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "americanair_conversations.csv")

BRAND = "AmericanAir"

COLS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]

_TS_FORMAT = "%a %b %d %H:%M:%S +0000 %Y"

# ---------------------------------------------------------------------------
# Step 1 - Ingest Dataset
# ---------------------------------------------------------------------------
print("=" * 65)
print("AmericanAir Customer-Support Conversation Reconstruction Pipeline")
print("=" * 65)
print(f"\n[1/6] Loading dataset: {INPUT_FILE}")

if not os.path.exists(INPUT_FILE):
    print(f"ERROR: File not found: {INPUT_FILE}", file=sys.stderr)
    sys.exit(1)

t0 = time.time()
df = pd.read_csv(INPUT_FILE, usecols=COLS, low_memory=False)
load_time = time.time() - t0

total_source_tweets = len(df)
print(f"      Loaded {total_source_tweets:,} rows in {load_time:.1f}s")

# Ensure tweet_id is valid integer
df["tweet_id"] = pd.to_numeric(df["tweet_id"], errors="coerce")
df.dropna(subset=["tweet_id"], inplace=True)
df["tweet_id"] = df["tweet_id"].astype(int)

# ---------------------------------------------------------------------------
# Step 2 - Build Reply Graph & Timestamp Cache
# ---------------------------------------------------------------------------
print("\n[2/6] Constructing reply graph and epoch timestamp index...")

# Build parent_map (child_id -> parent_id)
parent_map: dict[int, int] = {}

for tid, pid in zip(df["tweet_id"], df["in_response_to_tweet_id"]):
    if pd.notna(pid):
        try:
            parent_map[int(tid)] = int(float(pid))
        except (ValueError, OverflowError):
            pass

# Cross-reference response_tweet_id for missing inbound reply links
for parent_id, resp_field in zip(df["tweet_id"], df["response_tweet_id"]):
    if pd.notna(resp_field):
        for child_str in str(resp_field).split(","):
            child_str = child_str.strip()
            if child_str.isdigit():
                child_id = int(child_str)
                if child_id not in parent_map:
                    parent_map[child_id] = int(parent_id)

# Build children_map (parent_id -> [child_ids])
children_map: dict[int, list[int]] = defaultdict(list)
for child_id, par_id in parent_map.items():
    children_map[par_id].append(child_id)

# O(1) row lookup
tweet_lookup: dict[int, dict] = df.set_index("tweet_id").to_dict("index")

# Pre-parse timestamps into numeric epoch seconds for fast, accurate sorting
def parse_epoch(ts_str) -> float:
    if not ts_str:
        return float("inf")
    try:
        return datetime.strptime(str(ts_str), _TS_FORMAT).replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return float("inf")

tweet_epoch: dict[int, float] = {
    tid: parse_epoch(row.get("created_at"))
    for tid, row in tweet_lookup.items()
}

print(f"      Parent graph edges : {len(parent_map):,}")
print(f"      Unique parent nodes: {len(children_map):,}")

# ---------------------------------------------------------------------------
# Step 3 - Find Conversation Roots
# ---------------------------------------------------------------------------
print(f"\n[3/6] Identifying conversation trees involving {BRAND}...")

root_cache: dict[int, int] = {}

def find_root(start_id: int) -> int:
    """Finds top-most ancestor in the dataset using path compression & cycle guards."""
    path = []
    curr = start_id
    visited = set()

    while curr in parent_map:
        if curr in root_cache:
            curr = root_cache[curr]
            break
        if curr in visited:
            break
        visited.add(curr)
        path.append(curr)
        curr = parent_map[curr]

    for node in path:
        root_cache[node] = curr
    root_cache[start_id] = curr
    return curr

aa_mask = df["author_id"] == BRAND
aa_tweet_ids = set(df.loc[aa_mask, "tweet_id"])
aa_tweet_count = len(aa_tweet_ids)

inbound_mask = df["inbound"] == True  # noqa: E712
inbound_to_aa = df[inbound_mask & df["in_response_to_tweet_id"].isin(aa_tweet_ids)]
customer_reply_count = len(inbound_to_aa)

all_roots: set[int] = set()
for tid in aa_tweet_ids:
    all_roots.add(find_root(tid))
for tid in inbound_to_aa["tweet_id"]:
    all_roots.add(find_root(tid))

print(f"      {BRAND} authored tweets     : {aa_tweet_count:,}")
print(f"      Customer tweets -> {BRAND}: {customer_reply_count:,}")
print(f"      Distinct tree roots       : {len(all_roots):,}")

# ---------------------------------------------------------------------------
# Step 4 - Deconstruct Trees into Customer-Isolated Conversations
# ---------------------------------------------------------------------------
print("\n[4/6] Decomposing reply trees into customer-isolated conversations...")

reconstructed_conversations: list[list[int]] = []
assigned_tweets: set[int] = set()

for root_id in sorted(all_roots):
    # Collect all reachable nodes in this tree
    tree_nodes: list[int] = []
    queue: deque[int] = deque([root_id])
    visited_tree: set[int] = {root_id}

    while queue:
        curr = queue.popleft()
        tree_nodes.append(curr)
        for child in children_map.get(curr, []):
            if child not in visited_tree:
                visited_tree.add(child)
                queue.append(child)

    present_nodes = [tid for tid in tree_nodes if tid in tweet_lookup]
    if not present_nodes:
        continue

    # Find distinct customer authors in this tree
    customers_in_tree = {
        tweet_lookup[tid]["author_id"]
        for tid in present_nodes
        if tweet_lookup[tid]["inbound"] == True and tweet_lookup[tid]["author_id"] != BRAND
    }

    if not customers_in_tree:
        continue

    has_brand = any(tweet_lookup[tid]["author_id"] == BRAND for tid in present_nodes)
    if not has_brand:
        continue

    if len(customers_in_tree) == 1:
        # Case A: Tree has exactly 1 customer -> Single unified conversation
        cust_id = next(iter(customers_in_tree))
        # Keep strictly messages by this customer or AmericanAir
        conv_tweets = [
            tid for tid in present_nodes
            if tid not in assigned_tweets and tweet_lookup[tid]["author_id"] in (cust_id, BRAND)
        ]
        if conv_tweets:
            authors = {tweet_lookup[tid]["author_id"] for tid in conv_tweets}
            has_cust = any(tweet_lookup[tid]["author_id"] == cust_id for tid in conv_tweets)
            has_aa = BRAND in authors
            if has_cust and has_aa:
                conv_tweets.sort(key=lambda tid: (tweet_epoch.get(tid, float("inf")), tid))
                for tid in conv_tweets:
                    assigned_tweets.add(tid)
                reconstructed_conversations.append(conv_tweets)

    else:
        # Case B: Multi-customer tree (e.g. broadcast or bystander replies).
        # Decompose into separate conversations for each customer.
        for cust_id in sorted(customers_in_tree):
            cust_tweets = [
                tid for tid in present_nodes
                if tweet_lookup[tid]["author_id"] == cust_id and tid not in assigned_tweets
            ]
            if not cust_tweets:
                continue

            cust_conv_set: set[int] = set(cust_tweets)

            # Collect AA parent/replies directly associated with this customer
            for c_tid in cust_tweets:
                # Direct parent AA tweet if not a multi-customer broadcast
                par = parent_map.get(c_tid)
                if par and par in tweet_lookup and tweet_lookup[par]["author_id"] == BRAND:
                    if par not in assigned_tweets:
                        # Check how many distinct customers replied to this parent
                        par_cust_replies = {
                            tweet_lookup[ch]["author_id"]
                            for ch in children_map.get(par, [])
                            if ch in tweet_lookup and tweet_lookup[ch]["inbound"] == True
                        }
                        if len(par_cust_replies) <= 1:
                            cust_conv_set.add(par)

                # Direct downward replies to this customer's tweets
                q_down: deque[int] = deque([c_tid])
                while q_down:
                    curr_down = q_down.popleft()
                    for ch in children_map.get(curr_down, []):
                        if ch in tweet_lookup and ch not in assigned_tweets:
                            ch_author = tweet_lookup[ch]["author_id"]
                            if ch_author in (BRAND, cust_id):
                                cust_conv_set.add(ch)
                                q_down.append(ch)

            conv_tweets = [
                tid for tid in cust_conv_set
                if tid not in assigned_tweets and tweet_lookup[tid]["author_id"] in (cust_id, BRAND)
            ]
            if conv_tweets:
                authors = {tweet_lookup[tid]["author_id"] for tid in conv_tweets}
                has_cust = any(tweet_lookup[tid]["author_id"] == cust_id for tid in conv_tweets)
                has_aa = BRAND in authors
                if has_cust and has_aa:
                    conv_tweets.sort(key=lambda tid: (tweet_epoch.get(tid, float("inf")), tid))
                    for tid in conv_tweets:
                        assigned_tweets.add(tid)
                    reconstructed_conversations.append(conv_tweets)

# ---------------------------------------------------------------------------
# Step 5 - Build Structured Output DataFrame
# ---------------------------------------------------------------------------
print("\n[5/6] Formatting structured output...")

os.makedirs(OUTPUT_DIR, exist_ok=True)

rows = []
for conversation_id, conv in enumerate(reconstructed_conversations, start=1):
    for position, tid in enumerate(conv, start=1):
        row = tweet_lookup[tid]
        rows.append({
            "conversation_id": conversation_id,
            "position":        position,
            "tweet_id":        tid,
            "author_id":       row["author_id"],
            "inbound":         row["inbound"],
            "created_at":      row["created_at"],
            "text":            row["text"],
        })

result = pd.DataFrame(rows)
result.to_csv(OUTPUT_FILE, index=False)
print(f"      Saved output to: {OUTPUT_FILE}")

# ---------------------------------------------------------------------------
# Step 6 - Verification Statistics & Summary
# ---------------------------------------------------------------------------
print("\n[6/6] Computing validation statistics...")

total_convs = result["conversation_id"].nunique()
total_msgs  = len(result)
lengths     = result.groupby("conversation_id")["position"].max()

aa_conv_count = result[result["author_id"] == BRAND]["conversation_id"].nunique()
pct_with_aa   = (aa_conv_count / total_convs * 100) if total_convs > 0 else 0.0

# Multi-customer check
multi_cust_convs = 0
for _, grp in result.groupby("conversation_id"):
    custs = {
        a for a in grp.loc[grp["inbound"] == True, "author_id"].unique()  # noqa: E712
        if a != BRAND
    }
    if len(custs) > 1:
        multi_cust_convs += 1

dup_tweet_ids = result["tweet_id"].duplicated().sum()

print("\n" + "=" * 65)
print("RECONSTRUCTION METRICS & VALIDATION REPORT")
print("=" * 65)
print(f"  Total source tweets in dataset           : {total_source_tweets:,}")
print(f"  {BRAND} authored tweets                  : {aa_tweet_count:,}")
print(f"  Customer tweets replying to {BRAND}      : {customer_reply_count:,}")
print(f"  Reconstructed conversations              : {total_convs:,}")
print(f"  Messages retained                        : {total_msgs:,}")
print()
print("  Conversation Length Statistics:")
print(f"    Min length                             : {lengths.min()}")
print(f"    25th percentile                        : {lengths.quantile(0.25):.0f}")
print(f"    Median length                          : {lengths.median():.0f}")
print(f"    Mean length                            : {lengths.mean():.2f}")
print(f"    75th percentile                        : {lengths.quantile(0.75):.0f}")
print(f"    Max length                             : {lengths.max()}")
print()
print(f"  Conversations with >= 1 {BRAND} response : {aa_conv_count:,} ({pct_with_aa:.1f}%)")
print(f"  Conversations with multiple customer IDs : {multi_cust_convs} (Target: 0)")
print(f"  Duplicate tweet IDs across dataset       : {dup_tweet_ids} (Target: 0)")
print("=" * 65)
