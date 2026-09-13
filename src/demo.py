"""
demo.py - Interactive Terminal Demo for American Airlines Support Agent

Provides a clean, terminal-based interface for interacting with the existing
SupportAgent pipeline. Reuses the existing model, retrieval index, and escalation
engine without modifying core logic or evaluation data.

Usage:
    python src/demo.py
"""

import os
import sys
import textwrap

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

from src.support_agent import SupportAgent

SEPARATOR_MAJOR = "=" * 60
SEPARATOR_MINOR = "-" * 60


def format_text(text: str, width: int = 70, indent: str = "") -> str:
    """Format multiline text nicely with word wrapping."""
    lines = text.split("\n")
    wrapped_lines = []
    for line in lines:
        if line.strip():
            wrapped_lines.append(
                textwrap.fill(line.strip(), width=width, initial_indent=indent, subsequent_indent=indent)
            )
        else:
            wrapped_lines.append("")
    return "\n".join(wrapped_lines)


def run_demo():
    print(SEPARATOR_MAJOR)
    print("AMERICAN AIRLINES AI SUPPORT AGENT - INTERACTIVE DEMO")
    print(SEPARATOR_MAJOR)
    print("Loading historical interactions and initializing agent...")

    try:
        agent = SupportAgent()
    except Exception as e:
        print(f"\n[Error] Failed to initialize SupportAgent: {e}")
        return

    print("Agent ready.")
    print("\nType any customer inquiry or tweet below.")
    print("Examples you can try:")
    print("  - 'What is the earliest time I can check in my bags for flight AA123?'")
    print("  - 'My flight was cancelled and I need a full cash refund of $450 immediately!'")
    print("  - 'Missed my connection in DFW because of late arrival. Put me on next flight.'")
    print("  - 'I left my wallet on flight AA 497 seat 12F in Phoenix, can someone check?'")
    print("  - 'I have 53k miles in my AAdvantage account, how do I redeem award flights?'")
    print("\nType 'exit' or 'quit' to exit.")
    print(SEPARATOR_MAJOR)

    while True:
        try:
            print()
            customer_msg = input("Customer: ").strip()

            if not customer_msg:
                continue

            if customer_msg.lower() in ["exit", "quit", "q"]:
                print("\nExiting American Airlines AI Support Agent demo. Goodbye!")
                break

            # Process customer message through the existing support agent pipeline
            result = agent.process(customer_message=customer_msg, conversation_context="")

            intent = result.get("intent", "UNKNOWN")
            confidence = result.get("intent_confidence", 0.0)
            decision = result.get("decision", "UNKNOWN")
            decision_reason = result.get("decision_reason", "")
            draft_reply = result.get("draft_reply", "")
            evidence_list = result.get("evidence", [])
            top1_sim = result.get("top1_similarity", 0.0)

            # Clean decision reason for display if prefixed
            clean_reason = decision_reason
            if " — " in clean_reason:
                clean_reason = clean_reason.split(" — ", 1)[1].strip()

            print(f"\n{SEPARATOR_MINOR}")
            print("INTENT")
            print(SEPARATOR_MINOR)
            print(f"{intent}  (Estimated Confidence: {confidence:.2f})")

            print(f"\n{SEPARATOR_MINOR}")
            print("DECISION")
            print(SEPARATOR_MINOR)
            if decision == "AUTO_HANDLE":
                print("AUTO_HANDLE  [Safe, informational query with aligned precedent]")
            else:
                print("ESCALATE     [Requires human agent / reservation / billing access]")

            print(f"\n{SEPARATOR_MINOR}")
            print("REASON")
            print(SEPARATOR_MINOR)
            print(format_text(clean_reason, width=70))

            print(f"\n{SEPARATOR_MINOR}")
            print("SUGGESTED RESPONSE")
            print(SEPARATOR_MINOR)
            print(format_text(draft_reply, width=70))

            print(f"\n{SEPARATOR_MINOR}")
            print("HISTORICAL EVIDENCE")
            print(SEPARATOR_MINOR)
            if evidence_list:
                for idx, ev in enumerate(evidence_list, 1):
                    sim = ev.get("similarity", 0.0)
                    h_intent = ev.get("historical_intent", "UNKNOWN")
                    h_cust = ev.get("historical_customer_message", "").replace("\n", " ").strip()
                    h_resp = ev.get("historical_brand_response", "").replace("\n", " ").strip()

                    print(f"\n{idx}. [Similarity: {sim:.4f} | Past Intent: {h_intent}]")
                    print(f"   Customer   : \"{h_cust[:120]}{'...' if len(h_cust) > 120 else ''}\"")
                    print(f"   AmericanAir: \"{h_resp[:140]}{'...' if len(h_resp) > 140 else ''}\"")
            else:
                print("No relevant historical interactions found.")

            print(f"\n{SEPARATOR_MINOR}")

        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting American Airlines AI Support Agent demo. Goodbye!")
            break
        except Exception as e:
            print(f"\n[Warning] An error occurred while processing message: {e}")
            print("Continuing demo loop...")


if __name__ == "__main__":
    run_demo()
