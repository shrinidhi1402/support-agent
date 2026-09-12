"""
create_golden_set.py
====================
Phase 3: Build the 200-example hand-labelled golden evaluation dataset and
comprehensive annotation labeling guide for AmericanAir customer support.

Generates:
1. data/evaluation/golden_set.csv (200 examples with metadata, context, candidate intent, blank final_intent)
2. data/evaluation/labeling_guide.md (complete 10-intent annotation guidelines with boundaries)
"""

import os
import re
import sys
import random
import pandas as pd

# Set deterministic seed
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Paths
CONV_FILE    = os.path.join("data", "processed", "americanair_conversations.csv")
EVAL_DIR     = os.path.join("data", "evaluation")
GOLDEN_FILE  = os.path.join(EVAL_DIR, "golden_set.csv")
GUIDE_FILE   = os.path.join(EVAL_DIR, "labeling_guide.md")

TAXONOMY = [
    "FLIGHT_DISRUPTION",
    "REBOOKING_AND_CHANGES",
    "BAGGAGE_ISSUES",
    "SEATS_AND_CABIN",
    "BOOKING_AND_RESERVATIONS",
    "REFUNDS_AND_PAYMENTS",
    "CHECKIN_AND_BOARDING",
    "LOYALTY_AND_AADVANTAGE",
    "CUSTOMER_SERVICE_COMPLAINT",
    "GENERAL_INQUIRY_AND_OTHER",
]

# Deterministic regex patterns to identify candidate signals
PATTERNS = {
    "FLIGHT_DISRUPTION": re.compile(
        r"\b(delay|delayed|delays|late|behind schedule|cancel|canceled|cancelled|"
        r"cancellation|cancellations|diverted|stranded|stuck|tarmac|ground stop|"
        r"mechanical|maintenance issue|hours late|sitting on the runway)\b",
        re.IGNORECASE,
    ),
    "REBOOKING_AND_CHANGES": re.compile(
        r"\b(change flight|change date|switch flight|rebook|rebooked|rebooking|"
        r"standby|reschedule|missed connection|miss my connection|alternate flight|"
        r"change my ticket|put me on|next flight out|different flight)\b",
        re.IGNORECASE,
    ),
    "BAGGAGE_ISSUES": re.compile(
        r"\b(bag|bags|baggage|luggage|suitcase|suitcases|carousel|lost bag|"
        r"checked bag|damaged bag|claim number|pack|packed|belongings|"
        r"bag drop|missing luggage|stolen bag|bag tag)\b",
        re.IGNORECASE,
    ),
    "SEATS_AND_CABIN": re.compile(
        r"\b(seat|seats|seating|aisle|window|middle seat|legroom|first class|"
        r"business class|main cabin|upgrade|upgrades|row|exit row|bulkhead|"
        r"seat assignment|assigned seat|seat together|separated seats)\b",
        re.IGNORECASE,
    ),
    "BOOKING_AND_RESERVATIONS": re.compile(
        r"\b(book|booking|booked|reservation|reservations|reserve|itinerary|"
        r"confirmation code|record locator|ticket purchase|conf#|confirmation|"
        r"spelling on ticket|name change|middle name|hold ticket)\b",
        re.IGNORECASE,
    ),
    "REFUNDS_AND_PAYMENTS": re.compile(
        r"\b(refund|refunded|reimbursement|reimburse|charge|charged|fee|fees|"
        r"overcharge|receipt|voucher|credit|money back|compensation|pay|payment|"
        r"transaction|cost|bill|billed|double charge|unauthorized charge)\b",
        re.IGNORECASE,
    ),
    "CHECKIN_AND_BOARDING": re.compile(
        r"\b(check in|check-in|checkin|boarding|board|boarded|boarding pass|"
        r"boarding group|priority boarding|gate check|tsa|security line|"
        r"mobile pass|kiosk|cant check in|online check in)\b",
        re.IGNORECASE,
    ),
    "LOYALTY_AND_AADVANTAGE": re.compile(
        r"\b(aadvantage|miles|mileage|loyalty|status|platinum|executive platinum|"
        r"gold|conciergekey|elite|award ticket|upgrade miles|frequent flyer|"
        r"loyalty account|member number|missing miles|eqm|eqs)\b",
        re.IGNORECASE,
    ),
    "CUSTOMER_SERVICE_COMPLAINT": re.compile(
        r"\b(rude|attitude|unhelpful|terrible service|worst service|poor service|"
        r"customer service|supervisor|manager|representative|agent|attendant|"
        r"disrespectful|unprofessional|incompetent|horrible experience|"
        r"lack of empathy|screaming at passengers|horrible staff)\b",
        re.IGNORECASE,
    ),
    "GENERAL_INQUIRY_AND_OTHER": re.compile(
        r"\b(flight status|schedule|wifi|pet|unaccompanied minor|carry on policy|"
        r"thank you|thanks|great job|kudos|shoutout|question|does flight|"
        r"what time|information|curious|wondering|plane type)\b",
        re.IGNORECASE,
    ),
}

def generate_labeling_guide():
    """Generates the labeling_guide.md specification document."""
    guide_content = r"""# AmericanAir Customer Support: 10-Intent Labeling Guide

This labeling guide provides unambiguous definitions, positive examples, negative boundaries, and disambiguation rules for hand-labeling the AmericanAir Golden Evaluation Dataset.

---

## Intent Taxonomy Overview

The taxonomy consists of 10 mutually exclusive categories tailored to airline customer-support interactions:

1. `FLIGHT_DISRUPTION`
2. `REBOOKING_AND_CHANGES`
3. `BAGGAGE_ISSUES`
4. `SEATS_AND_CABIN`
5. `BOOKING_AND_RESERVATIONS`
6. `REFUNDS_AND_PAYMENTS`
7. `CHECKIN_AND_BOARDING`
8. `LOYALTY_AND_AADVANTAGE`
9. `CUSTOMER_SERVICE_COMPLAINT`
10. `GENERAL_INQUIRY_AND_OTHER`

---

## Core Annotation Principle: Primary User Goal

When a customer message contains multiple topics (e.g. a delayed flight that led to missed bags or rude agents), assign the intent based on the **Primary Actionable Goal** of the user in that specific turn:
- If the customer is venting or asking for status about the delay itself $\rightarrow$ `FLIGHT_DISRUPTION`.
- If the customer is asking to be placed on a new flight or standby $\rightarrow$ `REBOOKING_AND_CHANGES`.
- If the customer is demanding money back, hotel vouchers, or fee waivers $\rightarrow$ `REFUNDS_AND_PAYMENTS`.
- If the customer is seeking to locate missing bags caused by the delay $\rightarrow$ `BAGGAGE_ISSUES`.
- If the customer is escalating about how staff treated them during the delay $\rightarrow$ `CUSTOMER_SERVICE_COMPLAINT`.

---

## Detailed Intent Specifications

### 1. FLIGHT_DISRUPTION
- **Definition:** The customer reports, complains about, or inquires about an operational disruption to a flight (delay, cancellation, diversion, mechanical issue, tarmac hold, or weather hold).
- **Positive Examples:**
  - *"Why has AA1234 been delayed for over 3 hours in Charlotte?"*
  - *"Stuck on the tarmac for 90 minutes. No updates from the cockpit."*
  - *"Our flight just got canceled due to maintenance. What is happening?"*
- **Do NOT Use When:**
  - The customer is explicitly asking to be rebooked onto another flight $\rightarrow$ `REBOOKING_AND_CHANGES`.
  - The customer is requesting cash compensation, hotel vouchers, or meal expenses $\rightarrow$ `REFUNDS_AND_PAYMENTS`.
  - The customer is asking for routine schedule information for an on-time flight $\rightarrow$ `GENERAL_INQUIRY_AND_OTHER`.

---

### 2. REBOOKING_AND_CHANGES
- **Definition:** The customer wants to change an existing flight, rebook following a missed connection or cancellation, switch travel dates, or join a standby list.
- **Positive Examples:**
  - *"My flight was late and I missed my connection in DFW. Can you put me on the 7 PM flight?"*
  - *"Need to change my return flight from Miami to Sunday instead of Saturday."*
  - *"Can I get added to the standby list for the earlier flight to Boston?"*
- **Do NOT Use When:**
  - The customer is booking a brand new reservation $\rightarrow$ `BOOKING_AND_RESERVATIONS`.
  - The customer is just expressing anger about the delay without asking for alternative flights $\rightarrow$ `FLIGHT_DISRUPTION`.
  - The customer is seeking a refund for the canceled ticket $\rightarrow$ `REFUNDS_AND_PAYMENTS`.

---

### 3. BAGGAGE_ISSUES
- **Definition:** Issues involving physical checked or carry-on luggage: delayed/lost bags, damaged suitcases, baggage carousel waits, baggage tracing, or baggage policies.
- **Positive Examples:**
  - *"Arrived in Chicago but my bag is still in Dallas. Where is it?"*
  - *"My suitcase came off the carousel with a broken wheel and torn zipper."*
  - *"Waiting at baggage claim 4 for over an hour. Still no bags."*
- **Do NOT Use When:**
  - The customer is only disputing a $25/$30 checked bag charge on their credit card statement $\rightarrow$ `REFUNDS_AND_PAYMENTS`.
  - The customer is complaining about carrying bags down the jet bridge during boarding $\rightarrow$ `CHECKIN_AND_BOARDING`.

---

### 4. SEATS_AND_CABIN
- **Definition:** Inquiries and requests regarding seat assignments, cabin upgrades, seat selection (aisle/window), family seat separation, legroom, exit rows, or onboard physical seating comfort.
- **Positive Examples:**
  - *"My wife and I were seated in separate rows. Can you seat us together?"*
  - *"Are there any first class upgrade seats available on flight 240?"*
  - *"I selected an aisle seat but was reassigned to a middle seat at the gate."*
- **Do NOT Use When:**
  - The upgrade is being redeemed specifically through AAdvantage mileage or systemwide upgrades and the inquiry is about loyalty balance $\rightarrow$ `LOYALTY_AND_AADVANTAGE`.
  - The complaint is about flight attendant service while seated $\rightarrow$ `CUSTOMER_SERVICE_COMPLAINT`.

---

### 5. BOOKING_AND_RESERVATIONS
- **Definition:** Inquiries regarding creating new bookings, confirming reservation details, finding a lost record locator / PNR, correcting passenger name spellings, or general itinerary lookups.
- **Positive Examples:**
  - *"Can you resend the confirmation email for record locator ABCDEF?"*
  - *"I misspelled my daughter's middle name on our reservation. Can you fix it?"*
  - *"Trying to book a multi-city trip on your website and getting an error."*
- **Do NOT Use When:**
  - The customer wants to change an existing confirmed flight date $\rightarrow$ `REBOOKING_AND_CHANGES`.
  - The customer is inquiring about payment receipt or billing charge $\rightarrow$ `REFUNDS_AND_PAYMENTS`.

---

### 6. REFUNDS_AND_PAYMENTS
- **Definition:** Financial transactions, reimbursement requests, ticket refunds, disputed fees, travel vouchers, trip credits, or compensation for flight delays/damages.
- **Positive Examples:**
  - *"My flight was canceled and I did not travel. How do I get a full refund?"*
  - *"I was charged twice for seat selection on my credit card."*
  - *"Are you going to reimburse my hotel stay after the overnight cancellation in Philly?"*
- **Do NOT Use When:**
  - The customer is just asking what fares cost before booking $\rightarrow$ `GENERAL_INQUIRY_AND_OTHER`.
  - The customer wants to rebook without discussing money or refunds $\rightarrow$ `REBOOKING_AND_CHANGES`.

---

### 7. CHECKIN_AND_BOARDING
- **Definition:** Day-of-travel processes including web/app check-in errors, digital boarding passes, boarding groups, priority lane access, gate procedures, and airport security/TSA checkpoints.
- **Positive Examples:**
  - *"The app won't let me check in for my flight 24 hours in advance."*
  - *"Can't download my mobile boarding pass to Apple Wallet."*
  - *"Why are Group 8 passengers boarding before Group 4 at Gate C12?"*
- **Do NOT Use When:**
  - The gate agent was rude or yelled at passengers $\rightarrow$ `CUSTOMER_SERVICE_COMPLAINT`.
  - The flight is delayed at the gate $\rightarrow$ `FLIGHT_DISRUPTION`.

---

### 8. LOYALTY_AND_AADVANTAGE
- **Definition:** The AAdvantage frequent flyer program: elite status (Gold/Platinum/Executive Platinum), mileage accrual, missing miles claims, award tickets, account login, and club membership.
- **Positive Examples:**
  - *"My miles from last week's flight to London never posted to my account."*
  - *"How many more Elite Qualifying Miles do I need for Executive Platinum?"*
  - *"Unable to log into my AAdvantage account online. Password reset isn't working."*
- **Do NOT Use When:**
  - A passenger mentions their status casually while asking for a flight rebooking $\rightarrow$ `REBOOKING_AND_CHANGES`.
  - A passenger complains about an agent being rude to a Platinum member $\rightarrow$ `CUSTOMER_SERVICE_COMPLAINT`.

---

### 9. CUSTOMER_SERVICE_COMPLAINT
- **Definition:** Escalations and complaints regarding rude, disrespectful, unprofessional, or unhelpful staff behavior (gate agents, flight attendants, phone reps), poor in-flight service, or lack of communication.
- **Positive Examples:**
  - *"The gate agent in Miami was extremely rude and rolled her eyes when I asked a simple question."*
  - *"Flight attendant refused to provide water during a 4-hour flight. Horrible attitude."*
  - *"Waited on hold for 3 hours and the representative hung up on me."*
- **Do NOT Use When:**
  - The customer is frustrated by a flight delay but describes no staff misconduct $\rightarrow$ `FLIGHT_DISRUPTION`.
  - The customer is asking for compensation for a damaged bag $\rightarrow$ `BAGGAGE_ISSUES`.

---

### 10. GENERAL_INQUIRY_AND_OTHER
- **Definition:** Non-transactional inquiries, flight status checks for on-time flights, airline policies (pets, oxygen, minors, carry-on limits), compliments/kudos, and conversational tweets.
- **Positive Examples:**
  - *"What aircraft type operates flight AA100 from JFK to LHR?"*
  - *"Huge shoutout to Captain Dave and crew on flight 512 for a wonderful flight!"*
  - *"Can I bring a small cat in a carrier under the seat on a domestic flight?"*
  - *"Good morning @AmericanAir! Have a great Tuesday!"*
- **Do NOT Use When:**
  - The flight inquiry is reporting a severe delay $\rightarrow$ `FLIGHT_DISRUPTION`.
  - The policy question is specifically about checked bag fees already billed $\rightarrow$ `REFUNDS_AND_PAYMENTS`.

---

## Ambiguity Resolution Table

| Customer Says | Key Ambiguity | Resolution Rule |
| :--- | :--- | :--- |
| *"Delayed 4 hours, missed my cruise, I want my money back."* | Disruption vs Refund | `REFUNDS_AND_PAYMENTS` (The actionable demand is financial recovery) |
| *"Flight canceled, need to get to Dallas tonight for a wedding."* | Disruption vs Rebooking | `REBOOKING_AND_CHANGES` (The actionable demand is alternative transport) |
| *"Stuck on tarmac 2 hours and flight attendant was screaming at us."* | Disruption vs Staff Complaint | `CUSTOMER_SERVICE_COMPLAINT` (Specific interpersonal misconduct escalation) |
| *"Can I upgrade my seat to first class using my AAdvantage miles?"* | Seats vs Loyalty | `SEATS_AND_CABIN` (Physical seat upgrade is the primary objective) |
| *"Checked in on the app but it didn't give me my assigned seat."* | Checkin vs Seats | `CHECKIN_AND_BOARDING` (App check-in failure is the primary failure mode) |
"""
    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(GUIDE_FILE, "w", encoding="utf-8") as f:
        f.write(guide_content.strip() + "\n")
    print(f"Generated Labeling Guide at: {GUIDE_FILE}")


def build_golden_set():
    """Builds the 200-example golden dataset with stratification and context."""
    print(f"Loading conversation dataset from: {CONV_FILE}")
    df = pd.read_csv(CONV_FILE, low_memory=False)

    # Pre-calculate conversation length and preceding context
    conv_lengths = df.groupby("conversation_id")["position"].max().to_dict()

    # Pre-compile context mapping: (conv_id, position) -> context string
    # Context format: preceding messages in the same conversation
    print("Building conversation context lookup...")
    context_map = {}
    for conv_id, group in df.groupby("conversation_id"):
        sorted_group = group.sort_values("position")
        history = []
        for _, row in sorted_group.iterrows():
            pos = int(row["position"])
            author = "AmericanAir" if row["author_id"] == "AmericanAir" else f"Customer ({row['author_id']})"
            text = str(row["text"]).replace("\n", " ").strip()
            
            if not history:
                context_map[(conv_id, pos)] = "[No preceding context - conversation start]"
            else:
                context_map[(conv_id, pos)] = " | ".join(history)
            
            history.append(f"[{pos}] {author}: \"{text}\"")

    # Filter to customer messages only
    cust_df = df[df["inbound"] == True].copy()
    print(f"Customer messages available: {len(cust_df):,}")

    # Categorize candidates using deterministic heuristics
    records = []
    for _, row in cust_df.iterrows():
        cid = int(row["conversation_id"])
        pos = int(row["position"])
        tid = int(row["tweet_id"])
        text = str(row["text"]).strip()
        conv_len = conv_lengths.get(cid, 2)

        # Match against patterns
        matched = []
        for intent_name, pat in PATTERNS.items():
            if pat.search(text):
                matched.append(intent_name)

        # Assign candidate intent heuristic
        if not matched:
            candidate = "GENERAL_INQUIRY_AND_OTHER"
            difficulty = "easy"
            notes = "Clear general/conversational inquiry; no specific operational issue keyword."
        elif len(matched) == 1:
            candidate = matched[0]
            if pos >= 3 and len(text.split()) < 8:
                difficulty = "medium"
                notes = "Short mid-thread customer follow-up; context is essential to disambiguate."
            else:
                difficulty = "easy"
                notes = f"Unambiguous single-intent signal matching {candidate}."
        elif len(matched) == 2:
            candidate = matched[0]
            difficulty = "medium"
            notes = f"Dual-intent overlap: touches both {matched[0]} and {matched[1]}."
        else:
            candidate = matched[0]
            difficulty = "hard"
            notes = f"Multi-intent ambiguity spanning 3+ domains: {', '.join(matched)}."

        # Difficulty override for interesting cases
        if "crucial" in text.lower() or "disappointed" in text.lower() or "ridiculous" in text.lower():
            if difficulty == "easy":
                difficulty = "medium"

        # Determine conversation length bucket
        if conv_len == 2:
            len_bucket = "2_msgs"
        elif conv_len in (3, 4):
            len_bucket = "3_4_msgs"
        else:
            len_bucket = "5_plus_msgs"

        records.append({
            "conversation_id": cid,
            "position": pos,
            "tweet_id": tid,
            "customer_message": text,
            "conversation_context": context_map.get((cid, pos), "[No preceding context]"),
            "conv_length": conv_len,
            "len_bucket": len_bucket,
            "matched_count": len(matched),
            "matched_themes": matched,
            "candidate_intent": candidate,
            "difficulty": difficulty,
            "notes": notes,
        })

    all_pool = pd.DataFrame(records)
    print(f"Candidate pool built: {len(all_pool):,} records.")

    # Stratified Sampling:
    # Target: exactly 20 examples per candidate intent = 200 total examples.
    # Within each intent, balance:
    #   - Difficulty: easy (~10-12), medium (~5-7), hard (~3-5)
    #   - Length buckets: 2_msgs (~6-8), 3_4_msgs (~6-8), 5_plus_msgs (~5-7)
    selected_examples = []
    
    # We want ~30-40 total hard examples across the 200 (approx 3-4 per intent).
    for intent in TAXONOMY:
        intent_pool = all_pool[all_pool["candidate_intent"] == intent].copy()
        
        # Shuffle deterministically
        intent_pool = intent_pool.sample(frac=1.0, random_state=RANDOM_SEED)

        intent_selected = []

        # Target quota: 4 hard, 6 medium, 10 easy
        hard_pool   = intent_pool[intent_pool["difficulty"] == "hard"]
        med_pool    = intent_pool[intent_pool["difficulty"] == "medium"]
        easy_pool   = intent_pool[intent_pool["difficulty"] == "easy"]

        # Sample up to quotas
        take_hard = min(len(hard_pool), 4)
        if take_hard > 0:
            intent_selected.extend(hard_pool.head(take_hard).to_dict("records"))

        remaining = 20 - len(intent_selected)
        take_med = min(len(med_pool), min(6, remaining))
        if take_med > 0:
            intent_selected.extend(med_pool.head(take_med).to_dict("records"))

        remaining = 20 - len(intent_selected)
        take_easy = min(len(easy_pool), remaining)
        if take_easy > 0:
            intent_selected.extend(easy_pool.head(take_easy).to_dict("records"))

        # If still short of 20 (e.g. rare class), fill with remaining unused
        if len(intent_selected) < 20:
            used_tids = {x["tweet_id"] for x in intent_selected}
            fallback = intent_pool[~intent_pool["tweet_id"].isin(used_tids)]
            needed = 20 - len(intent_selected)
            intent_selected.extend(fallback.head(needed).to_dict("records"))

        # Keep exactly 20
        intent_selected = intent_selected[:20]

        # Verify balanced length representation where possible
        selected_examples.extend(intent_selected)

    # Sort final dataset deterministically by candidate intent then conversation_id
    final_df = pd.DataFrame(selected_examples)
    final_df = final_df.sort_values(["candidate_intent", "conversation_id", "position"]).reset_index(drop=True)

    # Assign example_id from 1 to 200
    final_df["example_id"] = range(1, len(final_df) + 1)
    final_df["final_intent"] = ""  # Left intentionally blank for human hand-labeling

    # Keep only requested columns
    output_cols = [
        "example_id",
        "conversation_id",
        "tweet_id",
        "customer_message",
        "conversation_context",
        "candidate_intent",
        "final_intent",
        "difficulty",
        "notes",
    ]
    golden_set = final_df[output_cols].copy()

    # Save golden_set.csv
    golden_set.to_csv(GOLDEN_FILE, index=False, encoding="utf-8")
    print(f"\nSaved Golden Dataset to: {GOLDEN_FILE}")

    # Print summary statistics
    print("\n" + "=" * 65)
    print("GOLDEN EVALUATION DATASET SUMMARY REPORT")
    print("=" * 65)
    print(f"Total Selected Examples: {len(golden_set)}")
    
    print("\n[A] Examples per Candidate Intent:")
    for intent, count in golden_set["candidate_intent"].value_counts().items():
        print(f"  - {intent:28s} : {count:3d} examples")

    print("\n[B] Examples per Difficulty Level:")
    for diff, count in golden_set["difficulty"].value_counts().items():
        pct = (count / len(golden_set)) * 100
        print(f"  - {diff.capitalize():8s} : {count:3d} ({pct:.1f}%)")

    print("\n[C] Conversation Length Distribution (of source conversations):")
    len_dist = final_df["len_bucket"].value_counts()
    for bucket, count in len_dist.items():
        pct = (count / len(final_df)) * 100
        print(f"  - {bucket:12s} : {count:3d} ({pct:.1f}%)")

    print("\n[D] Context Availability:")
    has_ctx = (golden_set["conversation_context"] != "[No preceding context - conversation start]").sum()
    print(f"  - Messages with preceding conversation context : {has_ctx} ({has_ctx/len(golden_set)*100:.1f}%)")
    print(f"  - Initial conversation-opening messages        : {len(golden_set) - has_ctx} ({(len(golden_set)-has_ctx)/len(golden_set)*100:.1f}%)")
    print("=" * 65)

    # Show 3 sample rows
    print("\n[E] Sample Golden Set Rows:")
    for _, row in golden_set.sample(3, random_state=RANDOM_SEED).iterrows():
        print("-" * 65)
        print(f"Example ID        : {row['example_id']}")
        print(f"Conversation ID   : {row['conversation_id']}")
        print(f"Tweet ID          : {row['tweet_id']}")
        print(f"Candidate Intent  : {row['candidate_intent']}")
        print(f"Final Intent      : '{row['final_intent']}' (BLANK for human labeling)")
        print(f"Difficulty        : {row['difficulty']}")
        print(f"Preceding Context : {row['conversation_context'][:100]}...")
        print(f"Customer Message  : \"{row['customer_message']}\"")
        print(f"Notes             : {row['notes']}")

def main():
    generate_labeling_guide()
    build_golden_set()

if __name__ == "__main__":
    main()
