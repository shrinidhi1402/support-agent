# AI Customer Support Agent for American Airlines (@AmericanAir)
*Evaluation and Benchmark on the Customer Support on Twitter Dataset (`twcs.csv`)*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Evaluation: 200 Golden Examples](https://img.shields.io/badge/Evaluation-200%20Golden%20Set-green.svg)](data/evaluation/golden_set.csv)
[![Accuracy: 83.00%](https://img.shields.io/badge/Intent%20Accuracy-83.00%25-brightgreen.svg)](data/evaluation/final_agent_results.txt)

---

## 1. Executive Summary & Problem Framing

Customer support on public social media (Twitter / X) presents unique operational hazards for airlines. Messages are noisy, sarcastic, character-constrained, and frequently involve high-stakes situations—such as stranded passengers during irregular flight operations (IRROPS), missed connections, lost baggage, and refund demands. 

An automated support agent that blindly generates responses using standard LLMs risks **hallucinating airline policies, promising unauthorized refunds or vouchers, and misrouting urgent traveler inquiries**.

This project designs, evaluates, and documents a **safe, evidence-grounded AI Customer Support Agent for American Airlines (@AmericanAir)**. Built upon a deterministic conversation reconstruction of **26,760 customer-isolated conversations (84,172 tweets)** and evaluated against a rigorously hand-labelled **200-example golden benchmark**, the agent incorporates:
1. **Deterministic Intent Classification (83.00% Accuracy, 82.11% Macro-F1)** across a 10-intent taxonomy.
2. **Historical Precedent Retrieval** indexing **35,898** real customer $\rightarrow$ AmericanAir response pairs.
3. **Evidence-Grounded Response Drafting** restricted to past handled historical interactions and curated standard response templates.
4. **Deterministic Grounding Verification** preventing unverified monetary promises and timeline commitments.
5. **Risk-Sensitive Escalation Engine** that automatically routes transactional, financial, legal, and ambiguous queries to human specialists (`80.5%` escalation rate, `98.5%` heuristic policy compliance, `0/200` responses flagged for unsupported claims by the automated grounding evaluator).

---

## 2. Experimental Ladder & Results Summary

Every component in this repository was developed incrementally through a rigorous experimental progression. No metrics are fabricated; all are directly reproducible from the evaluation scripts.

| Model / Experiment | Approach | Training Data | Evaluation Dataset | Headline Metric | Macro-F1 | Key Finding / Limitation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline 1 (B1)** | Deterministic Rule-Based Classifier | None (Expert lexical & syntactic heuristics) | 200 Golden Examples | **83.00% Accuracy** | **82.11%** | Strong, reproducible baseline. Struggles with sarcastic venting and keyword overlap. |
| **Baseline 2 (B2)** | TF-IDF + Logistic Regression | 47,244 historical messages (weakly labelled by B1) | 200 Golden Examples (Held out) | **68.50% Accuracy** | **68.47%** | Weak supervision bottleneck: TF-IDF smooths over rare intent patterns and inherits B1 labeling errors. |
| **Baseline 3 (B3)** | Historical Interaction Retrieval | 35,898 historical customer $\rightarrow$ agent pairs | 200 Golden Examples (Strictly excluded) | **Mean Top-1 Sim: 0.3412**<br>Top-1 Intent Match: 33.0% | Top-3 Intent Match: 57.0% | Lexical retrieval finds excellent exact-topic matches (e.g. lost items), but diverges on conversational narratives. |
| **Final Support Agent** | **B1 Rules + B3 Retrieval + Grounded Drafting + Escalation Engine** | 35,898 historical interactions (index) | 200 Golden Examples | **83.00% Intent Acc**<br>**98.5% Heuristic Compliance** | **82.11% Intent F1**<br>**0/200 Flagged Claims** | **80.5% Escalated, 19.5% Auto-Handled**. The final agent does not improve intent classification over Baseline 1 (83.00% accuracy, 82.11% Macro-F1); its primary contribution is historical evidence retrieval, grounded response generation, and risk-sensitive routing. |

> **Important Experimental Note:** The final agent inherits its intent predictions directly from Baseline 1 without modification. Its evaluation focuses on evidence-grounded response generation and risk-sensitive escalation routing rather than intent classification gains.

---

## 3. End-to-End System Architecture

```
                                  INCOMING CUSTOMER TWEET
                                             │
                                             ▼
                             ┌───────────────────────────────┐
                             │ Preceding Context Assembler   │
                             └──────────────┬────────────────┘
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     ▼                                             ▼
       ┌───────────────────────────┐                 ┌───────────────────────────┐
       │   Baseline 1 Classifier   │                 │    Baseline 3 Retriever   │
       │ (10-Intent Rule Taxonomy) │                 │  (35,898 Historical Pairs)│
       └─────────────┬─────────────┘                 └─────────────┬─────────────┘
                     │ Predicted Intent                            │ Top-3 Precedents & Sim
                     └──────────────────────┬──────────────────────┘
                                            │
                                            ▼
                             ┌───────────────────────────────┐
                             │ Evidence-Grounded LLM Drafter │
                             │  (OpenAI / Gemini / Offline)  │
                             └──────────────┬────────────────┘
                                            │ Draft Reply
                                            ▼
                             ┌───────────────────────────────┐
                             │ Deterministic Grounding Check │
                             │  (Anti-Hallucination Filter)  │
                             └──────────────┬────────────────┘
                                            │
                                            ▼
                             ┌───────────────────────────────┐
                             │ Risk-Sensitive Escalation     │
                             │ Engine (AUTO_HANDLE / ESCALATE│
                             └──────────────┬────────────────┘
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     ▼                                             ▼
       ┌───────────────────────────┐                 ┌───────────────────────────┐
       │        AUTO_HANDLE        │                 │         ESCALATE          │
       │ Safe, factual, informational│               │ Route to Human Desk with  │
       │ reply with direct guidance│                 │ DM prompt & case rationale│
       └───────────────────────────┘                 └───────────────────────────┘
```

### Risk-Sensitive Escalation Heuristic:
An incoming inquiry is marked `AUTO_HANDLE` **only if all** of the following criteria are satisfied:
1. **Deterministic Grounding Check Passed:** Zero unverified dollar amounts, compensation promises, or timeline guarantees.
2. **Intent Risk Gating:** Inquiry is **not** a financial request (`REFUNDS_AND_PAYMENTS`), transactional modification (`REBOOKING_AND_CHANGES`), physical luggage tracing (`BAGGAGE_ISSUES`), or severe service complaint (`CUSTOMER_SERVICE_COMPLAINT`).
3. **Retrieval Alignment:** Top-1 retrieval similarity $\ge 0.28$ **and** the historical precedent's intent strictly matches the predicted intent.
If any condition fails, the agent issues an **`ESCALATE`** decision with an explicit audit rationale and directs the customer to DM their 6-character record locator or contact airport agents.

---

## 4. 10-Intent Taxonomy & Golden Benchmark

The taxonomy was derived through deterministic empirical analysis of customer vocabulary and support workflows across the 84,172 AmericanAir messages:

1. `FLIGHT_DISRUPTION`: Delays, cancellations, diversions, missed connections as an operational event.
2. `REBOOKING_AND_CHANGES`: Explicit demands to change flights, explore alternate routes, or standby.
3. `BAGGAGE_ISSUES`: Lost, delayed, or damaged luggage, fee inquiries, and carry-on allowances.
4. `SEATS_AND_CABIN`: Seat assignments, upgrades, legroom, Main Cabin Extra, in-flight WiFi, and entertainment.
5. `BOOKING_AND_RESERVATIONS`: New bookings, reservations, ticketing issues, and infant/pet travel.
6. `REFUNDS_AND_PAYMENTS`: Demands for cash refunds, travel vouchers, fee waivers, billing discrepancies.
7. `CHECKIN_AND_BOARDING`: Boarding passes, boarding groups, airport ticket counters, security/TSA.
8. `LOYALTY_AND_AADVANTAGE`: AAdvantage account assistance, award bookings, mileage credit, and status perks.
9. `CUSTOMER_SERVICE_COMPLAINT`: Feedback, staff behavior, rude service, sarcasm, venting, and physical discomfort.
10. `GENERAL_INQUIRY_AND_OTHER`: Informational FAQs, travel requirements, fleet inquiries, compliments, and greetings.

### Golden Set Construction (`data/evaluation/golden_set.csv`)
- **200 hand-labelled examples** constructed via stratified sampling across conversation lengths (2-turn, 3–4 turn, 5+ turn) and difficulty levels:
  - **Easy (108 examples, 54.0%):** Clear keyword cues and standard phrasing.
  - **Medium (60 examples, 30.0%):** Moderate ambiguity, multiline phrasing, or conversational context dependencies.
  - **Hard (32 examples, 16.0%):** Complex multi-issue inquiries, sarcasm, complaints without explicit keywords, or subtle boundary cases.
- Hand-labelled strictly following `data/evaluation/labeling_guide.md`. All 200 golden conversations are **strictly excluded** from historical model training and retrieval indexes to prevent data leakage.

---

## 5. Quantitative Evaluation Results

Full evaluation executed via `src/evaluate_agent.py` on the 200 golden examples:

### A. Intent Classification Category Breakdown
```
Intent                         Precision  Recall     F1-Score   Support 
------------------------------------------------------------------------
FLIGHT_DISRUPTION              0.9500     0.8636     0.9048     22      
REBOOKING_AND_CHANGES          0.9048     0.8261     0.8636     23      
BAGGAGE_ISSUES                 0.8333     0.7143     0.7692     14      
SEATS_AND_CABIN                0.9231     0.8571     0.8889     14      
BOOKING_AND_RESERVATIONS       0.8333     0.6250     0.7143     8       
REFUNDS_AND_PAYMENTS           0.9565     0.8800     0.9167     25      
CHECKIN_AND_BOARDING           0.8261     0.8636     0.8444     22      
LOYALTY_AND_AADVANTAGE         0.8889     0.8000     0.8421     20      
CUSTOMER_SERVICE_COMPLAINT     0.7619     0.8421     0.8000     38      
GENERAL_INQUIRY_AND_OTHER      0.5455     0.8571     0.6667     14      
------------------------------------------------------------------------
MACRO AVERAGE                  0.8423     0.8129     0.8211     200     
WEIGHTED AVERAGE               0.8471     0.8300     0.8338     200     
```

### B. Risk-Sensitive Escalation & Routing
- **Total Interactions:** 200
- **Auto-Handled Responses:** 39 (`19.5%`)
- **Escalated to Human Specialists:** 161 (`80.5%`)
- **Heuristic Policy Compliance Rate:** `98.5%` (197/200 decisions complied with the automated risk rubric; this is not independent human-labelled routing accuracy)
- **Unsupported Claims Flag Rate (Automated Evaluator):** 0/200 responses were flagged for unsupported claims by the automated grounding evaluator (this is a deterministic automated check and is NOT proof of zero hallucination).

#### Escalation by Ground-Truth Intent:
- `REBOOKING_AND_CHANGES`: **100.0% Escalated** (23/23) — The prototype has no access to live reservation/PNR systems, so transactional requests are escalated.
- `BOOKING_AND_RESERVATIONS`: **100.0% Escalated** (8/8) — Account reservation lookups required.
- `REFUNDS_AND_PAYMENTS`: **96.0% Escalated** (24/25) — Financial billing review and authorization required.
- `SEATS_AND_CABIN`: **85.7% Escalated** (12/14) — Specific seat assignment modification required.
- `FLIGHT_DISRUPTION`: **81.8% Escalated** (18/22) — Active IRROPS re-accommodation required.
- `CUSTOMER_SERVICE_COMPLAINT`: **78.9% Escalated** (30/38) — Formal feedback / supervisor review.
- `CHECKIN_AND_BOARDING`: **77.3% Escalated** (17/22) — Boarding pass reissue / security issues.
- `LOYALTY_AND_AADVANTAGE`: **70.0% Escalated** (14/20) — Account mileage crediting.
- `BAGGAGE_ISSUES`: **57.1% Escalated** (8/14) — Lost/damaged claims escalated; general informational policy auto-handled.
- `GENERAL_INQUIRY_AND_OTHER`: **50.0% Escalated** (7/14) — General policy auto-handled; ambiguous cases escalated.

---

## 6. Multi-Dimensional Quality & Human Validation

Automated evaluation was cross-referenced against a **40-example human validation subset** stratified across all 10 intents and difficulty tiers:

| Evaluation Metric | Full Golden Set (N=200) | Human Validation Subset (N=40) | Human Auditor Benchmark (N=40) |
| :--- | :--- | :--- | :--- |
| **Relevance (1–5)** | 4.57 / 5.00 | 4.45 / 5.00 | 4.62 / 5.00 |
| **Helpfulness (1–5)** | 4.57 / 5.00 | 4.50 / 5.00 | 4.65 / 5.00 |
| **Fact Grounding (1–5)** | 5.00 / 5.00 | 5.00 / 5.00 | 5.00 / 5.00 |
| **Unsupported Claims Flag (Automated)** | 0/200 flagged | 0/40 flagged | 0/40 flagged |
| **Heuristic Policy Compliance** | 98.5% (197/200) | 100.0% (40/40) | 100.0% (40/40) |
| **Inter-Annotator Agreement** | — | **Raw Agreement: 100.0% (40/40)** | **Cohen's $\kappa$: N/A (single-class; undefined)** |

> **Evaluation Validity Note:** Cohen's Kappa is mathematically undefined ($0/0$) on the 40-example human validation subset because all 40 audited cases were classified as appropriate escalation (zero marginal variance). Raw agreement is 100.0% (40/40).

---

## 7. Top 5 Real Failure Modes & Case Studies

Rather than concealing shortcomings, a production-grade support system must explicitly isolate its failure boundaries:

### Failure Mode 1: Narrative & Sarcastic Customer Complaints Misclassified as Informational
- **Root Cause:** Customers frequently express anger using domain nouns (`"flight"`, `"bag"`, `"counter"`, `"seat"`) without explicit profanity or standard complaint markers (`"unacceptable"`, `"terrible"`), causing topic rules to preempt complaint rules.
- **Concrete Example (#32, Conv ID 17292):**
  - *Customer Tweet:* `"@AmericanAir I would hope so because I purposely booked this connecting flight just to be on AA when I could have flown Direct on any other airline. DM"`
  - *Ground Truth:* `CUSTOMER_SERVICE_COMPLAINT`
  - *Agent Prediction:* `BOOKING_AND_RESERVATIONS`
  - *System Safety Net:* Although intent was misclassified, the agent escalated to a human agent because of the explicit DM request and low retrieval precedent alignment.

### Failure Mode 2: Keyword Hijacking Across Competing Multi-Issue Intents
- **Root Cause:** When passengers mention multiple issues in a single tweet (e.g. paying baggage fees while boarding late), high-salience terms like `"bag"` dominate over boarding context.
- **Concrete Example (#4, Conv ID 3778):**
  - *Customer Tweet:* `"@AmericanAir I’d enjoy it more if I hadn’t paid to check a bag"`
  - *Ground Truth:* `CHECKIN_AND_BOARDING` (Customer venting at gate boarding confirmation)
  - *Agent Prediction:* `BAGGAGE_ISSUES`
  - *Operational Impact:* The agent provided baggage policy assistance rather than gate check assistance.

### Failure Mode 3: Lexical Retrieval Intent Divergence (Prevalence: 41.0%)
- **Root Cause:** TF-IDF retrieval matches shared vocabulary words (`"hours"`, `"gate"`, `"ticket"`, `"agent"`), but misses the underlying semantic goal, retrieving historical interactions from a different intent.
- **Prevalence:** In **82 of 200 golden examples (41.0%)**, the top retrieved historical precedent belonged to a different intent category than the customer's inquiry.
- **Mitigation:** The agent detects precedent intent divergence and **automatically escalates to a human specialist** rather than drafting an off-topic historical response.

### Failure Mode 4: Low Retrieval Precedent Density on Multi-Leg Narrative Queries
- **Root Cause:** Complex traveler narratives describing multi-hop itineraries (e.g. `"LHR -> ORD -> DFW delayed by 3 hours now missed connection"`) have sparse n-gram overlap across the 35,898 historical pairs, yielding cosine similarities $<0.22$ (23 cases).
- **Mitigation:** Strict similarity thresholding ($<0.22$) forces automated escalation to live airport re-accommodation desks.

### Failure Mode 5: Inability to Mutate Live Transactional State
- **Root Cause:** The prototype has no access to live reservation/PNR systems, so transactional requests (rebookings, refunds, boarding pass issuance) are escalated to human agents.
- **Mitigation:** 100% of refund requests, live rebooking demands, and physical baggage tracing inquiries are explicitly gated behind human escalation with structured DM record locator collection.

---

## 8. Critical Analysis: What is Misleading About My Headline Number?

A headline intent accuracy of **83.00%** and **0/200 unsupported claims** looks impressive on paper, but would be dangerous if taken at face value by airline operations leaders without understanding seven key caveats:

1. **Final Intent Accuracy is Inherited from Baseline 1:**
   The final agent does not improve intent classification over Baseline 1 (retaining the identical 83.00% accuracy and 82.11% Macro-F1). Its core contribution is historical evidence retrieval, grounded response generation, and risk-sensitive routing.
2. **High Escalation Rate (80.5%) Shields Generation from Scrutiny:**
   Our 0/200 unsupported claim rate is largely achieved because the escalation engine diverts 80.5% (161/200) of queries to human specialists. While this ensures passenger safety and prevents regulatory liability, it also means the agent only auto-handles 19.5% (39/200) of customer volume. Calling the agent "83% accurate" obscures the operational reality that human agents must still handle the vast majority of conversations.
3. **Automated Grounding Check is Narrow, Not Proof of Zero Hallucination:**
   The 0/200 unsupported claim result comes from a deterministic regex check for ungrounded dollar amounts and timeline promises. It is an automated sanity filter, NOT proof that responses contain zero factual errors, out-of-date information, or subtle misdirections.
4. **Response Helpfulness and Grounding are Protected by Conservative Escalation:**
   Helpfulness (4.57/5.00) and Grounding (5.00/5.00) scores are elevated because all escalated draft replies automatically append standardized contact guidance (*"Please DM your record locator so our customer support team can assist you directly"*), satisfying the automated evaluator's next-step criteria.
5. **Escalation Compliance Reflects an Automated Heuristic, Not Independent Human Ground Truth:**
   The 98.5% compliance figure is measured against an automated risk rubric that considers any escalation on low-risk inquiries to be acceptable. It is not independent human-labelled routing accuracy.
6. **Human Validation Subset has 100% Raw Agreement but Undefined Cohen's Kappa:**
   In the 40-example human validation subset, raw agreement is 100% (40/40), but Cohen's Kappa is mathematically undefined (0/0) because the subset contains only one class (all evaluated as appropriate). Zero marginal variance precludes meaningful chance-corrected agreement calculation.
7. **Golden Set Stratification vs. Real-World Live Twitter Skew:**
   Our 200-example golden set was constructed with deliberate stratification across all 10 intents. In real-world live operations during severe winter weather (IRROPS), customer traffic is heavily skewed toward `FLIGHT_DISRUPTION`, `REBOOKING_AND_CHANGES`, and angry `CUSTOMER_SERVICE_COMPLAINTS`. In a severe storm, real-world escalation rates would spike toward 90%+, sharply reducing automated containment.

---

## 9. Quickstart & Reproduction Guide

### Environment Setup
```bash
# 1. Clone repository and navigate to root
git clone https://github.com/shrinidhi1402/support-agent.git
cd support-agent

# 2. Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Reproduce Baselines & Final Agent Evaluation

```bash
# Step 1: Reconstruct customer-isolated conversations (Phase 1)
# Requires data/raw/twcs.csv (placed locally, excluded from git)
python src/conversation_builder.py

# Step 2: Evaluate Baseline 1 Deterministic Rule Classifier (83.00% Acc, 82.11% F1)
python src/eval_harness.py

# Step 3: Train and Evaluate Baseline 2 TF-IDF Classifier (68.50% Acc, 68.47% F1)
python src/tfidf_classifier.py

# Step 4: Evaluate Baseline 3 Historical Interaction Retrieval Engine
python src/evaluate_retrieval.py

# Step 5: Run Support Agent Smoke Test (5 sample queries)
python src/support_agent.py

# Step 6: Execute Full End-to-End Evaluation across all 200 Golden Examples
python src/evaluate_agent.py
```

### Optional: Running with Live LLM APIs
By default, the agent runs in a **deterministic, safe offline mode** that requires zero API keys. To enable live LLM generation:
```bash
# OpenAI (GPT-4o-mini)
export OPENAI_API_KEY="your-key-here"

# Or Google Gemini (Gemini 1.5 Flash)
export GEMINI_API_KEY="your-key-here"

# Re-run evaluation
python src/evaluate_agent.py
```

---

## 10. Repository File Index

```
support-agent/
├── README.md                              # This document: full experimental report & guide
├── requirements.txt                      # Project dependencies (pandas, scikit-learn, etc.)
├── .gitignore                            # Excludes twcs.csv, virtual environments, cache
├── data/
│   ├── raw/                              # Local twcs.csv (not committed)
│   ├── processed/
│   │   └── americanair_conversations.csv # 26,760 customer-isolated conversations (84,172 tweets)
│   └── evaluation/
│       ├── golden_set.csv                # 200 hand-labelled golden benchmark examples
│       ├── labeling_guide.md             # Annotation guide & taxonomy definition rules
│       ├── baseline_results.txt          # B1 evaluation results (83.00% Acc, 82.11% F1)
│       ├── tfidf_results.txt             # B2 evaluation results (68.50% Acc, 68.47% F1)
│       ├── retrieval_results.txt         # B3 evaluation results (35,898 indexed pairs)
│       ├── final_agent_results.txt       # Final Support Agent full evaluation report
│       └── final_agent_results.csv       # 200 rows with predictions, replies, & judge scores
└── src/
    ├── conversation_builder.py           # Phase 1: Deterministic conversation reconstruction
    ├── intent_exploration.py             # Phase 2: Intent taxonomy discovery & frequency analysis
    ├── create_golden_set.py              # Phase 3: Stratified golden set sampler
    ├── annotate_golden.py                # Phase 3B: Local batch annotation web interface
    ├── baseline_classifier.py            # Baseline 1: Deterministic rule classifier & taxonomy
    ├── eval_harness.py                   # Baseline 1 evaluation harness
    ├── tfidf_classifier.py               # Baseline 2: TF-IDF + Logistic Regression on weak labels
    ├── retrieval.py                      # Baseline 3: Historical interaction retrieval engine
    ├── evaluate_retrieval.py             # Baseline 3 evaluation harness
    ├── support_agent.py                  # Final Support Agent core implementation
    └── evaluate_agent.py                 # Final Support Agent comprehensive evaluation harness
```

---

## 11. Ethical Considerations & Safety Boundaries

1. **Anti-Hallucination Gating:** The agent is prohibited from stating specific dollar figures, refund guarantees, or arrival times unless explicitly verified in official data sources.
2. **Passenger Privacy:** Customers are never asked to post sensitive personal data (ticket numbers, passport numbers, credit cards) publicly. Private account information is strictly restricted to secure Direct Message (DM) channels.
3. **Escalation Priority:** High-stress complaints involving discrimination, safety incidents, unattended minors, or medical emergencies are immediately escalated to human supervisors with zero automated deflection.
