"""
intent_exploration.py
=====================
Phase 2: Intent Discovery & Taxonomy Design for AmericanAir Support.

Analyzes customer messages (inbound == True) from:
    data/processed/americanair_conversations.csv

Objectives:
1. Ingest customer messages and compute descriptive metrics.
2. Clean noise (mentions, URLs, stopwords, punctuation).
3. Compute vocabulary frequency (unigrams, bigrams).
4. Perform deterministic theme discovery across major aviation support domains.
5. Identify cross-theme ambiguity and multi-intent overlaps.
6. Recommend a practical, production-ready intent taxonomy (8-12 intents)
   grounded in empirical tweet distributions with clear boundary definitions.
7. Save the full structured report to:
    data/processed/intent_exploration.txt
"""

import os
import re
import sys
from collections import Counter
from io import StringIO

import pandas as pd

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Configuration & File Paths
# ---------------------------------------------------------------------------
INPUT_FILE  = os.path.join("data", "processed", "americanair_conversations.csv")
OUTPUT_FILE = os.path.join("data", "processed", "intent_exploration.txt")

# Standard English stopwords
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves", "just", "get", "got", "like", "us", "u", "ur",
    "now", "one", "also", "even", "still", "aa", "americanair", "amp"
}

# ---------------------------------------------------------------------------
# Theme Keyword / Regex Patterns for Deterministic Exploration
# ---------------------------------------------------------------------------
THEME_PATTERNS = {
    "Flight Delay & Cancellation": re.compile(
        r"\b(delay|delayed|delays|late|behind schedule|cancel|canceled|cancelled|"
        r"cancellation|cancellations|diverted|stranded|stuck|missed connection|"
        r"hours late|wait time|waiting for hours|held up)\b",
        re.IGNORECASE,
    ),
    "Baggage & Luggage": re.compile(
        r"\b(bag|bags|baggage|luggage|suitcase|suitcases|carousel|lost bag|claim|"
        r"checked bag|damaged bag|pack|packed|belongings|lost luggage|bag drop)\b",
        re.IGNORECASE,
    ),
    "Booking & Reservations": re.compile(
        r"\b(book|booking|booked|reservation|reservations|reserve|itinerary|"
        r"confirmation code|record locator|ticket purchase|conf#|confirmation)\b",
        re.IGNORECASE,
    ),
    "Payments, Refunds & Fees": re.compile(
        r"\b(refund|refunded|reimbursement|reimburse|charge|charged|fee|fees|"
        r"overcharge|receipt|voucher|credit|money back|compensation|pay|payment|"
        r"transaction|cost|bill|billed)\b",
        re.IGNORECASE,
    ),
    "Ticket Change & Rebooking": re.compile(
        r"\b(change flight|change date|switch flight|rebook|rebooked|rebooking|"
        r"standby|reschedule|change my ticket|modify reservation|change fee|"
        r"flight change|alternate flight)\b",
        re.IGNORECASE,
    ),
    "Seats & Cabin Comfort": re.compile(
        r"\b(seat|seats|seating|aisle|window|middle seat|legroom|first class|"
        r"business class|main cabin|upgrade|upgrades|row|exit row|bulkhead|"
        r"seat assignment|assigned seat)\b",
        re.IGNORECASE,
    ),
    "Check-in & Boarding": re.compile(
        r"\b(check in|check-in|checkin|boarding|board|boarded|boarding pass|"
        r"boarding group|priority boarding|gate check|tsa|security line|"
        r"mobile pass|kiosk)\b",
        re.IGNORECASE,
    ),
    "Loyalty & AAdvantage": re.compile(
        r"\b(aadvantage|miles|mileage|loyalty|status|platinum|executive platinum|"
        r"gold|conciergekey|elite|award ticket|upgrade miles|frequent flyer|"
        r"loyalty account|member number)\b",
        re.IGNORECASE,
    ),
    "Airport & Gate Operations": re.compile(
        r"\b(gate|terminal|airport|tarmac|runway|ramp|ground crew|maintenance|"
        r"mechanical|pilot|plane|aircraft|deicing|gate agent)\b",
        re.IGNORECASE,
    ),
    "Flight Information & Status": re.compile(
        r"\b(flight status|schedule|on time|arrival|departure|eta|etd|track|"
        r"flight info|when does|flight number|is flight|arrive|depart)\b",
        re.IGNORECASE,
    ),
    "Customer Service & Staff Complaint": re.compile(
        r"\b(rude|attitude|unhelpful|terrible service|worst service|poor service|"
        r"customer service|supervisor|manager|representative|agent|attendant|"
        r"disrespectful|unprofessional|incompetent|horrible experience)\b",
        re.IGNORECASE,
    ),
    "Pricing & Fares": re.compile(
        r"\b(price|pricing|fare|fares|expensive|cheap|sale|sales|discount|"
        r"deal|deals|quote|rate|rates|price drop|black friday)\b",
        re.IGNORECASE,
    ),
}

def clean_text_for_vocab(text: str) -> list[str]:
    """Removes mentions, URLs, punctuation, and stopwords for vocabulary profiling."""
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    words = text.lower().split()
    return [w for w in words if len(w) > 2 and w not in STOPWORDS and not w.isdigit()]

def safe_str(text: str) -> str:
    """Encodes safely for console printing across diverse platforms."""
    return str(text).encode("ascii", errors="replace").decode("ascii")

def main():
    buffer = StringIO()

    def p(text=""):
        try:
            print(text)
        except UnicodeEncodeError:
            print(safe_str(text))
        buffer.write(text + "\n")

    p("=" * 75)
    p("PHASE 2: INTENT DISCOVERY & TAXONOMY EXPLORATION REPORT")
    p("Brand: AmericanAir | Dataset: Customer-Support on Twitter")
    p("=" * 75)

    if not os.path.exists(INPUT_FILE):
        p(f"ERROR: Dataset not found at: {INPUT_FILE}")
        sys.exit(1)

    # 1. Ingest Data
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    cust_df = df[df["inbound"] == True].copy()

    total_msgs = len(cust_df)
    total_convs = df["conversation_id"].nunique()
    unique_custs = cust_df["author_id"].nunique()

    p("\n[1] DATASET OVERVIEW & HIGH-LEVEL METRICS")
    p("-" * 75)
    p(f"  Total conversations reconstructed   : {total_convs:,}")
    p(f"  Total messages (all participants)   : {len(df):,}")
    p(f"  Customer messages (inbound == True) : {total_msgs:,}")
    p(f"  Unique customer author IDs          : {unique_custs:,}")
    p(f"  Average customer messages / conv    : {total_msgs / total_convs:.2f}")

    # Message Length Metrics
    char_lens = cust_df["text"].str.len()
    word_lens = cust_df["text"].str.split().str.len()

    p("\n[2] CUSTOMER MESSAGE LENGTH PROFILE")
    p("-" * 75)
    p(f"  Character length : min={char_lens.min()}, 25%={char_lens.quantile(0.25):.0f}, "
      f"median={char_lens.median():.0f}, mean={char_lens.mean():.1f}, "
      f"75%={char_lens.quantile(0.75):.0f}, max={char_lens.max()}")
    p(f"  Word count       : min={word_lens.min()}, 25%={word_lens.quantile(0.25):.0f}, "
      f"median={word_lens.median():.0f}, mean={word_lens.mean():.1f}, "
      f"75%={word_lens.quantile(0.75):.0f}, max={word_lens.max()}")

    # 3. Vocabulary Profiling
    p("\n[3] VOCABULARY & PHRASE FREQUENCY (Twitter Noise Removed)")
    p("-" * 75)
    all_tokens = []
    bigrams = []

    for text in cust_df["text"].dropna():
        tokens = clean_text_for_vocab(text)
        all_tokens.extend(tokens)
        if len(tokens) >= 2:
            bigrams.extend([f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)])

    unigram_counts = Counter(all_tokens).most_common(25)
    bigram_counts = Counter(bigrams).most_common(20)

    p("  Top 25 Most Frequent Unigrams:")
    for rank, (word, count) in enumerate(unigram_counts, start=1):
        p(f"    {rank:2d}. {word:15s} : {count:6,d} occurrences")

    p("\n  Top 20 Most Frequent Bigrams (Phrases):")
    for rank, (phrase, count) in enumerate(bigram_counts, start=1):
        p(f"    {rank:2d}. {phrase:22s} : {count:5,d} occurrences")

    # 4. Representative Sample Customer Messages
    p("\n[4] REPRESENTATIVE CUSTOMER MESSAGE EXAMPLES")
    p("-" * 75)
    sample_rows = cust_df.sample(n=5, random_state=42)
    for idx, (_, row) in enumerate(sample_rows.iterrows(), start=1):
        clean_preview = str(row["text"]).replace("\n", " ")
        p(f"  Ex {idx} [Conv {row['conversation_id']} | Author {row['author_id']}]:")
        p(f"       \"{clean_preview}\"")

    # 5. Deterministic Support Theme Matching
    p("\n[5] DETERMINISTIC SUPPORT THEME DISCOVERY & DISTRIBUTION")
    p("-" * 75)
    p("  * Note: Regex/keyword matching is an exploratory instrument to discover")
    p("    latent intent volume, NOT a production classifier.")

    theme_matches = {theme: [] for theme in THEME_PATTERNS}
    multi_theme_records = []

    for _, row in cust_df.iterrows():
        matched_themes = []
        text = str(row["text"])
        for theme_name, pattern in THEME_PATTERNS.items():
            if pattern.search(text):
                matched_themes.append(theme_name)
                theme_matches[theme_name].append(row)

        multi_theme_records.append({
            "conversation_id": row["conversation_id"],
            "tweet_id": row["tweet_id"],
            "text": text,
            "themes": matched_themes,
            "theme_count": len(matched_themes),
        })

    # Sort themes by volume
    theme_counts = {t: len(matches) for t, matches in theme_matches.items()}
    sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)

    p("\n  Domain Volume Ranking:")
    for rank, (theme_name, count) in enumerate(sorted_themes, start=1):
        pct = (count / total_msgs) * 100
        p(f"    {rank:2d}. {theme_name:36s} : {count:6,d} ({pct:5.1f}%)")

    # Representative Examples per Domain
    p("\n  Domain Representative Examples (5 per theme):")
    for theme_name, _ in sorted_themes:
        p(f"\n  >>> THEME: {theme_name.upper()} (Total matches: {len(theme_matches[theme_name]):,})")
        matches = theme_matches[theme_name][:5]
        for ex_idx, row in enumerate(matches, start=1):
            txt = str(row["text"]).replace("\n", " ")[:160]
            p(f"      [{ex_idx}] (Conv {row['conversation_id']} | Tweet {row['tweet_id']})")
            p(f"          \"{txt}...\"")

    # 6. Multi-Intent Overlap & Ambiguity Analysis
    p("\n[6] AMBIGUITY & CROSS-INTENT OVERLAP ANALYSIS")
    p("-" * 75)

    multi_df = pd.DataFrame(multi_theme_records)
    count_dist = multi_df["theme_count"].value_counts().sort_index()

    p("  Number of Themes Matched per Message:")
    for num_themes, count in count_dist.items():
        pct = (count / total_msgs) * 100
        label = "No specific keyword match (General/Other)" if num_themes == 0 else f"{num_themes} theme(s)"
        p(f"    {label:44s} : {count:6,d} ({pct:5.1f}%)")

    # Frequent Co-occurrences
    pair_counter = Counter()
    for themes in multi_df["themes"]:
        if len(themes) >= 2:
            sorted_t = sorted(themes)
            for i in range(len(sorted_t)):
                for j in range(i + 1, len(sorted_t)):
                    pair_counter[(sorted_t[i], sorted_t[j])] += 1

    p("\n  Top 10 Most Frequent Multi-Intent Co-occurrences:")
    for rank, ((t1, t2), count) in enumerate(pair_counter.most_common(10), start=1):
        pct = (count / total_msgs) * 100
        p(f"    {rank:2d}. {t1} + {t2}")
        p(f"        Overlap count: {count:,} ({pct:.1f}% of all customer tweets)")

    p("\n  Representative Ambiguous / Multi-Intent Customer Messages:")
    ambiguous_samples = multi_df[multi_df["theme_count"] >= 3].head(6)
    for idx, (_, row) in enumerate(ambiguous_samples.iterrows(), start=1):
        themes_joined = " | ".join(row["themes"])
        p(f"\n  Ambiguous Ex {idx} [Matched Themes: {themes_joined}]:")
        p(f"      Text: \"{row['text'].replace(chr(10), ' ')}\"")
        p(f"      Ambiguity Analysis: The customer message bridges distinct operational units.")

    # 7. Candidate Taxonomy Recommendation
    p("\n" + "=" * 75)
    p("[7] RECOMMENDED CANDIDATE INTENT TAXONOMY (10 INTENTS)")
    p("=" * 75)
    p("""
Based on empirical data volume, operational actionability, and semantic distinctness,
we recommend a 10-intent taxonomy tailored specifically for AmericanAir support:

1. FLIGHT_DISRUPTION (Delay / Cancellation / Diversion)
   - Scope: Notifications and complaints regarding delayed, canceled, or diverted flights,
     runway ground stops, and stranded passengers.
   - Distinctness: Focuses on the status of the disrupted flight itself, distinct from
     the subsequent request to rebook or refund.

2. REBOOKING_AND_CHANGES (Flight Change / Missed Connection / Standby)
   - Scope: Actionable requests to modify flight dates, rebook due to missed connections,
     switch itinerary, or join standby lists.
   - Distinctness: While often triggered by a flight delay, this intent captures the
     forward-looking request to modify a ticket/route rather than merely reporting the delay.

3. BAGGAGE_ISSUES (Lost / Delayed / Damaged Luggage & Baggage Fees)
   - Scope: Inquiries regarding missing baggage, carousel wait times, damaged suitcases,
     baggage claim tracking, and checked bag fees.
   - Distinctness: Isolated to physical luggage lifecycle and baggage tracking.

4. SEATS_AND_CABIN (Seat Selection / Assignments / Cabin Upgrades)
   - Scope: Requests to change seat (aisle/window), complaints about seat separation,
     seat upgrade eligibility, bulkhead/exit rows, and onboard cabin comfort.
   - Distinctness: Focuses strictly on physical seating and cabin tier upgrades, distinct
     from general booking or reservation management.

5. BOOKING_AND_RESERVATIONS (New Booking / Confirmation / Itinerary)
   - Scope: Inquiries about booking flights, record locators/confirmation codes, name
     corrections, and schedule confirmations.
   - Distinctness: Handles standard booking inquiries and itinerary confirmations before
     any flight changes, cancellations, or refunds occur.

6. REFUNDS_AND_PAYMENTS (Refunds / Billing / Unwanted Charges / Vouchers)
   - Scope: Requests for monetary refunds, flight voucher redemption, compensation for
     disruptions, and disputes over unexpected credit card charges.
   - Distinctness: Financial transaction resolutions, distinct from ticket rebooking or
     initial fare shopping.

7. CHECKIN_AND_BOARDING (Boarding Passes / Mobile App / Gates / TSA)
   - Scope: Inability to check in on the website/app, mobile boarding pass errors, gate
     assignments, boarding group sequences, and TSA checkpoint delays.
   - Distinctness: Day-of-travel gate and departure readiness, distinct from overall
     flight delays or seat requests.

8. LOYALTY_AND_AADVANTAGE (Miles / Elite Status / Account Management)
   - Scope: Missing AAdvantage miles, tier status progress (Gold/Platinum/Exec Plat),
     mileage redemption, and account login issues.
   - Distinctness: Long-term customer loyalty program management, distinct from regular
     flight transactions.

9. CUSTOMER_SERVICE_COMPLAINT (Staff Conduct / Agent Attitude / In-Flight Service)
   - Scope: Escalations regarding rude gate agents, unhelpful phone representatives,
     in-flight crew behavior, and general service quality complaints.
   - Distinctness: Feedback on human interactions and service standards where no
     immediate logistical booking change is necessarily requested.

10. GENERAL_INQUIRY_AND_OTHER (Flight Status Inquiries / FAQs / Kudos / Miscellaneous)
    - Scope: Routine queries ("What plane flies AA100?"), compliments/thank you messages,
      travel policy queries (pets, unaccompanied minors), and unclassifiable tweets.
    - Distinctness: Catches unclassifiable or conversational noise that does not require
      a specialized transactional support workflow.
""")

    p("=" * 75)
    p("END OF REPORT")
    p("=" * 75)

    # Save to file
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(buffer.getvalue())

    p(f"\nReport successfully saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
