"""
support_agent.py - AI Customer Support Agent for AmericanAir

Combines:
1. Baseline 1 DeterministicRuleClassifier (intent classification)
2. Baseline 3 HistoricalInteractionRetriever (TF-IDF retrieval over 35,898 interactions)
3. Evidence-grounded LLM response generation (OpenAI / Gemini / Offline fallback)
4. Lightweight deterministic grounding validation
5. Risk-sensitive, interpretable escalation logic (AUTO_HANDLE vs. ESCALATE)
"""

import os
import sys
import re
import json
from typing import Dict, List, Any, Optional, Tuple
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

from src.baseline_classifier import DeterministicRuleClassifier, TAXONOMY
from src.retrieval import HistoricalInteractionRetriever

# -----------------------------------------------------------------------------
# LLM Provider Abstraction
# -----------------------------------------------------------------------------

class LLMClient:
    """Unified LLM client supporting OpenAI, Gemini, and offline fallback."""

    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None, api_key: Optional[str] = None):
        self.provider = provider or self._detect_provider()
        self.api_key = api_key or self._detect_api_key(self.provider)
        self.model = model or self._default_model(self.provider)
        self._init_client()

    def _detect_provider(self) -> str:
        if os.environ.get("OPENAI_API_KEY"):
            return "openai"
        if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            return "gemini"
        return "offline"

    def _detect_api_key(self, provider: str) -> Optional[str]:
        if provider == "openai":
            return os.environ.get("OPENAI_API_KEY")
        if provider == "gemini":
            return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        return None

    def _default_model(self, provider: str) -> str:
        if provider == "openai":
            return "gpt-4o-mini"
        if provider == "gemini":
            return "gemini-1.5-flash"
        return "offline-rule-generator"

    def _init_client(self):
        self.client = None
        if self.provider == "openai" and self.api_key:
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
            except Exception as e:
                print(f"[Warning] Failed to initialize OpenAI client: {e}. Falling back to offline mode.")
                self.provider = "offline"
        elif self.provider == "gemini" and self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel(self.model)
            except Exception as e:
                print(f"[Warning] Failed to initialize Gemini client: {e}. Falling back to offline mode.")
                self.provider = "offline"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response from LLM or offline fallback."""
        if self.provider == "openai" and self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                    max_tokens=300,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"[Warning] OpenAI API call failed: {e}. Using offline generator.")
                return self._offline_generate(user_prompt)

        elif self.provider == "gemini" and self.client:
            try:
                combined_prompt = f"{system_prompt}\n\nUser Request:\n{user_prompt}"
                response = self.client.generate_content(
                    combined_prompt,
                    generation_config={"temperature": 0.2, "max_output_tokens": 300},
                )
                return response.text.strip()
            except Exception as e:
                print(f"[Warning] Gemini API call failed: {e}. Using offline generator.")
                return self._offline_generate(user_prompt)

        return self._offline_generate(user_prompt)

    def _offline_generate(self, user_prompt: str) -> str:
        """
        Deterministic, safe offline response generator used when no external API key is active.
        Strictly synthesizes from top retrieved evidence or recommends escalation.
        """
        # Parse intent and evidence from user_prompt
        intent_match = re.search(r"Predicted Intent:\s*([A-Z_]+)", user_prompt)
        intent = intent_match.group(1) if intent_match else "GENERAL_INQUIRY_AND_OTHER"

        # Check for retrieved response and evidence intent
        evidence_matches = re.findall(
            r"\[Evidence #\d+\] \(Cosine Similarity: ([\d\.]+), Past Intent: ([A-Z_]+)\).*?Historical AmericanAir Response:\s*\"([^\"]+)\"",
            user_prompt,
            re.DOTALL,
        )

        if evidence_matches:
            top_sim, top_hist_intent, top_resp = evidence_matches[0]
            top_sim = float(top_sim)
            clean_resp = re.sub(r"^@\w+\s*", "", top_resp).strip()
            # If top evidence is reasonably similar and aligns in intent, use it
            if top_sim >= 0.28 and top_hist_intent == intent:
                return clean_resp

        # Safe, intent-grounded fallbacks when offline and no high-confidence aligned precedent exists
        intent_fallbacks = {
            "REFUNDS_AND_PAYMENTS": "For refund requests or receipt verification, please submit your request at prefts.aa.com or DM your 6-character record locator.",
            "BAGGAGE_ISSUES": "For delayed or damaged baggage, please file an incident report with airport Baggage Service or DM your 13-character bag tag number.",
            "FLIGHT_DISRUPTION": "We apologize for the flight disruption. Please check real-time departure options in the American Airlines app or DM your record locator for rebooking assistance.",
            "REBOOKING_AND_CHANGES": "To review alternate flights or make reservation changes, please DM your 6-character record locator so an agent can assist.",
            "SEATS_AND_CABIN": "For seat assignments or cabin inquiries, please manage your booking on aa.com or DM your record locator for seat options.",
            "CHECKIN_AND_BOARDING": "Online check-in opens 24 hours prior to departure. At the airport, ticket counters typically open 2-3 hours before departure. Please DM your flight details if you need specific airport counter hours.",
            "BOOKING_AND_RESERVATIONS": "For new bookings or reservation updates, please visit aa.com or DM your reservation details so we can assist.",
            "LOYALTY_AND_AADVANTAGE": "For AAdvantage account assistance, award bookings, or mileage credit, please DM your AAdvantage number so our loyalty team can help.",
            "CUSTOMER_SERVICE_COMPLAINT": "We hold our service to high standards and are sorry for your experience. Please DM your record locator and details so we can forward this to our Customer Relations team.",
            "GENERAL_INQUIRY_AND_OTHER": "Please DM your 6-character record locator and flight details so our team can look into this for you.",
        }
        return intent_fallbacks.get(intent, "Please DM your record locator so our team can assist you.")

# -----------------------------------------------------------------------------
# Support Agent Core
# -----------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an AI customer support agent for American Airlines (@AmericanAir) on Twitter.
Your role is to draft grounded, professional, and concise customer support replies based EXCLUSIVELY on historical evidence of how American Airlines previously handled similar issues.

CRITICAL OPERATIONAL RULES:
1. Grounding in Evidence:
   - Use ONLY information supported by the provided conversation and historical evidence examples.
   - The historical examples show past handling, NOT guaranteed current airline policy.
2. Anti-Hallucination & Policy Boundaries:
   - NEVER invent airline policies, baggage fees, refund amounts, or compensation promises.
   - NEVER promise specific meal/hotel vouchers or cash amounts unless explicitly verified in the customer's conversation.
   - NEVER fabricate flight schedules, departure times, gate assignments, or booking statuses.
3. Safe Escalation & Uncertainty:
   - If the historical evidence does not provide a clear, reliable way to address the customer's specific actionable request, recommend escalation.
   - If retrieved examples conflict or address different issues, do not guess; recommend direct agent escalation via DM or phone.
4. Tone & Style:
   - Be empathetic, concise, and professional (under 280 characters if possible, standard Twitter support format).
   - Never expose internal reasoning, cosine similarity, TF-IDF, or that an LLM was used.
   - When account details or private verification are needed, instruct the customer to DM their 6-character record locator."""

class SupportAgent:
    """
    Final Support Agent combining deterministic intent classification,
    historical retrieval, grounded LLM generation, and risk-sensitive escalation.
    """

    def __init__(
        self,
        retriever: Optional[HistoricalInteractionRetriever] = None,
        llm_client: Optional[LLMClient] = None,
        top_k: int = 3,
        sim_threshold: float = 0.22,
    ):
        self.rule_clf = DeterministicRuleClassifier()
        self.retriever = retriever or HistoricalInteractionRetriever(top_k=top_k)
        self.llm = llm_client or LLMClient()
        self.top_k = top_k
        self.sim_threshold = sim_threshold

    def process(
        self,
        customer_message: str,
        conversation_context: str = "",
    ) -> Dict[str, Any]:
        """
        Process incoming customer message through the full agent pipeline.
        Returns structured decision, evidence trail, and draft response.
        """
        # Step 1: Predict Intent using Baseline 1
        intent = self.rule_clf.predict(customer_message, conversation_context)

        # Step 2: Retrieve Top-k Historical Interactions
        evidence_items = self.retriever.retrieve(
            customer_message=customer_message,
            conversation_context=conversation_context,
            top_k=self.top_k,
        )

        top1_sim = evidence_items[0]["similarity_score"] if evidence_items else 0.0
        top3_sim = float(np.mean([e["similarity_score"] for e in evidence_items])) if evidence_items else 0.0

        # Estimate intent confidence based on lexical match and retrieval alignment
        top1_hist_intent = evidence_items[0]["historical_intent"] if evidence_items else None
        retrieval_intent_match = any(e["historical_intent"] == intent for e in evidence_items)
        
        if top1_hist_intent == intent and top1_sim >= 0.40:
            intent_confidence = 0.95
        elif retrieval_intent_match and top1_sim >= 0.25:
            intent_confidence = 0.85
        elif top1_sim >= 0.20:
            intent_confidence = 0.70
        else:
            intent_confidence = 0.55

        # Step 3: LLM Response Generation
        user_prompt = self._build_prompt(
            customer_message=customer_message,
            conversation_context=conversation_context,
            predicted_intent=intent,
            evidence_items=evidence_items,
        )

        draft_reply = self.llm.generate(SYSTEM_PROMPT, user_prompt)

        # Step 4: Deterministic Grounding Validation
        grounding_passed, grounding_issue = self._validate_grounding(
            draft_reply=draft_reply,
            evidence_items=evidence_items,
            customer_message=customer_message,
        )

        # Step 5: Risk-Sensitive Escalation Decision
        decision, decision_reason = self._evaluate_escalation(
            intent=intent,
            top1_sim=top1_sim,
            top3_sim=top3_sim,
            evidence_items=evidence_items,
            customer_message=customer_message,
            grounding_passed=grounding_passed,
            grounding_issue=grounding_issue,
            draft_reply=draft_reply,
        )

        # If escalated, ensure draft reply includes clear escalation guidance
        if decision == "ESCALATE" and not any(k in draft_reply.lower() for k in ["dm", "call", "agent", "desk", "representative", "contact"]):
            draft_reply = f"{draft_reply} Please DM your record locator so our customer support team can assist you directly."

        # Format Evidence Trail
        structured_evidence = []
        for e in evidence_items:
            structured_evidence.append({
                "conversation_id": e["conversation_id"],
                "similarity": round(e["similarity_score"], 4),
                "historical_intent": e["historical_intent"],
                "historical_customer_message": e["customer_message"],
                "historical_brand_response": e["agent_response"],
            })

        return {
            "intent": intent,
            "intent_confidence": round(intent_confidence, 2),
            "decision": decision,
            "decision_reason": decision_reason,
            "draft_reply": draft_reply,
            "evidence": structured_evidence,
            "top1_similarity": round(top1_sim, 4),
            "top3_similarity": round(top3_sim, 4),
            "retrieval_intent_match": retrieval_intent_match,
            "grounding_passed": grounding_passed,
        }

    def _build_prompt(
        self,
        customer_message: str,
        conversation_context: str,
        predicted_intent: str,
        evidence_items: List[Dict[str, Any]],
    ) -> str:
        """Construct structured user prompt containing context, intent, and evidence."""
        lines = []
        lines.append(f"Predicted Intent: {predicted_intent}")

        if conversation_context and "[No preceding" not in conversation_context:
            lines.append(f"Preceding Conversation Context:\n{conversation_context}")
        else:
            lines.append("Preceding Conversation Context: [Conversation start - first message]")

        lines.append(f"Customer's Current Message:\n\"{customer_message}\"")
        lines.append("\nTop Retrieved Historical AmericanAir Interactions (Past Evidence):")

        for idx, item in enumerate(evidence_items, 1):
            lines.append(f"\n[Evidence #{idx}] (Cosine Similarity: {item['similarity_score']:.4f}, Past Intent: {item['historical_intent']})")
            if item["context"]:
                lines.append(f"  Historical Context: \"{item['context'][:100]}...\"")
            lines.append(f"  Historical Customer Message : \"{item['customer_message']}\"")
            lines.append(f"  Historical AmericanAir Response: \"{item['agent_response']}\"")

        lines.append("\nInstructions:")
        lines.append("Draft a helpful, customer-support appropriate tweet response grounded strictly in the provided evidence.")
        lines.append("If the evidence does not provide a safe, verifiable resolution, politely direct the customer to DM their record locator or speak with an airport agent.")
        return "\n".join(lines)

    def _validate_grounding(
        self,
        draft_reply: str,
        evidence_items: List[Dict[str, Any]],
        customer_message: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Lightweight deterministic check for obvious unsupported claims:
        - Monetary dollar figures not in evidence or customer message
        - Fabricated timeline promises ('within 24 hours', 'by 5pm')
        - Absolute compensation/refund promises
        """
        reply_lower = draft_reply.lower()
        combined_source = customer_message.lower() + " " + " ".join([e["agent_response"].lower() for e in evidence_items])

        # 1. Check monetary amounts (e.g. $100, $25, 200 dollars)
        reply_money = re.findall(r"\$\s*\d+|\b\d+\s*dollars?\b", reply_lower)
        for m in reply_money:
            if m not in combined_source:
                return False, f"Unsupported monetary amount mentioned: '{m}'"

        # 2. Check firm compensation promises without source support
        promise_phrases = ["we will refund", "full refund guaranteed", "we will reimburse your hotel", "will send a voucher for"]
        for p in promise_phrases:
            if p in reply_lower and p not in combined_source:
                return False, f"Unsupported compensation promise: '{p}'"

        # 3. Check specific fabricated timeline guarantees
        timeline_phrases = ["guarantee arrival within", "will be credited in 24 hours", "refund will arrive in 48 hours"]
        for t in timeline_phrases:
            if t in reply_lower and t not in combined_source:
                return False, f"Unsupported timeline commitment: '{t}'"

        return True, None

    def _evaluate_escalation(
        self,
        intent: str,
        top1_sim: float,
        top3_sim: float,
        evidence_items: List[Dict[str, Any]],
        customer_message: str,
        grounding_passed: bool,
        grounding_issue: Optional[str],
        draft_reply: str,
    ) -> Tuple[str, str]:
        """
        Interpretable escalation decision logic:
        Decides AUTO_HANDLE vs. ESCALATE with clear human-readable explanation.
        """
        msg_lower = customer_message.lower()

        # Rule 1: Grounding failure
        if not grounding_passed:
            return "ESCALATE", f"ESCALATE — Grounding check detected unsupported claim: {grounding_issue}"

        # Rule 2: High-risk financial intents (REFUNDS_AND_PAYMENTS)
        if intent == "REFUNDS_AND_PAYMENTS":
            if any(k in msg_lower for k in ["refund", "reimburse", "compensat", "charge", "voucher", "fee", "pay back", "receipt"]):
                return "ESCALATE", "ESCALATE — Customer requests refund, fee waiver, or monetary compensation requiring authorized billing verification."

        # Rule 3: Direct operational rebooking / re-accommodation demands across any intent
        if re.search(r"\b(put me on|book me on|switch (me|flight|seat)|rebook|standby|alternate flight|next flight|missed\s+.*connection)\b", msg_lower):
            return "ESCALATE", "ESCALATE — Rebooking, standby changes, or missed connection re-accommodation requires live reservation agent access."

        # Rule 4: Very weak retrieval evidence
        if top1_sim < self.sim_threshold:
            return "ESCALATE", f"ESCALATE — Low historical retrieval similarity ({top1_sim:.4f} < {self.sim_threshold}); insufficient historical precedent to auto-handle safely."

        # Rule 5: Critical reservation/PNR modifications
        if intent in ["REBOOKING_AND_CHANGES", "BOOKING_AND_RESERVATIONS"]:
            if any(k in msg_lower for k in ["cancel my", "change my ticket", "change flight", "change reservation", "holding ticket"]):
                return "ESCALATE", "ESCALATE — Reservation modifications require direct agent access to the reservation system."

        # Rule 6: Safety, discrimination, legal threats, or severe complaints
        if intent == "CUSTOMER_SERVICE_COMPLAINT":
            if any(k in msg_lower for k in ["discriminate", "discrimination", "lawyer", "attorney", "police", "assault", "harass", "unacceptable", "disgrace", "sue"]):
                return "ESCALATE", "ESCALATE — Severe customer service complaint or legal/safety sensitivity requiring supervisor escalation."

        # Rule 7: Lost or damaged baggage requiring airport incident claim
        if intent == "BAGGAGE_ISSUES":
            if any(k in msg_lower for k in ["lost my bag", "missing luggage", "never arrived", "damaged suitcase", "broken wheel", "where is my bag"]):
                return "ESCALATE", "ESCALATE — Physical baggage tracing and damage claims require filing an airport claim with Baggage Services."

        # Rule 8: Severe flight disruptions requiring operational reaccommodation
        if intent == "FLIGHT_DISRUPTION":
            if any(k in msg_lower for k in ["stuck", "stranded", "diverted", "cancelled", "hours late", "delay"]) and any(k in msg_lower for k in ["help", "option", "hotel", "food", "voucher"]):
                return "ESCALATE", "ESCALATE — Active operational flight disruption requiring live operational assistance."

        # Rule 9: Safe Auto-handling requires BOTH sufficient similarity AND intent consistency
        retrieval_intents = [e["historical_intent"] for e in evidence_items]
        top1_hist_intent = retrieval_intents[0] if retrieval_intents else None

        if top1_sim >= 0.35 and top1_hist_intent == intent:
            return "AUTO_HANDLE", f"AUTO_HANDLE — Strong historical evidence (similarity {top1_sim:.4f}) aligns with historical precedent for {intent}."

        if top1_sim >= 0.28 and top1_hist_intent == intent:
            return "AUTO_HANDLE", f"AUTO_HANDLE — Retrieved evidence (similarity {top1_sim:.4f}) provides sufficient historical guidance for a standard public response."

        return "ESCALATE", f"ESCALATE — Ambiguous request or historical precedent intent divergence (retrieved {top1_hist_intent} vs predicted {intent}); directing customer to verified human support channel."

def main():
    print("Initializing SupportAgent...")
    agent = SupportAgent()
    print("Agent ready.")

    test_examples = [
        ("What is the earliest time I can check in my bags for flight AA123 at SJC?", ""),
        ("My flight was cancelled and I need a full cash refund of $450 immediately!", ""),
        ("Thank you so much to the gate agent in Charlotte for helping us board early!", ""),
        ("I left my wallet on flight AA 497 seat 12F in Phoenix, can someone check Lost and Found?", ""),
        ("Missed my connection in DFW because of the late arrival. Can you put me on the 8 PM flight?", ""),
    ]

    print("\n" + "=" * 80)
    print("SMOKE TEST: Running SupportAgent on 5 Sample Customer Messages")
    print("=" * 80)

    for i, (msg, ctx) in enumerate(test_examples, 1):
        print(f"\n>>> Test Case #{i}:")
        print(f"Customer Message: \"{msg}\"")
        res = agent.process(msg, ctx)
        print(f"Predicted Intent : {res['intent']} (Confidence: {res['intent_confidence']})")
        print(f"Decision         : {res['decision']}")
        print(f"Decision Reason  : {res['decision_reason']}")
        print(f"Draft Reply      : \"{res['draft_reply']}\"")
        print(f"Top-1 Similarity : {res['top1_similarity']}")
        print(f"Grounding Passed : {res['grounding_passed']}")

if __name__ == "__main__":
    main()
