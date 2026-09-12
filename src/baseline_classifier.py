"""
baseline_classifier.py - Deterministic Rule-Based Intent Classifier (Baseline 1)

Classifies AmericanAir customer messages into the 10-intent taxonomy using
primary actionable goal detection, regex patterns, and precedence rules.
"""

import re
from typing import Optional

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

def clean_text(text: Optional[str]) -> str:
    """Normalize text by lowercasing and stripping URLs/mentions."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    return text.strip()

class DeterministicRuleClassifier:
    """
    Baseline 1: Rule-based classifier with actionable goal precedence hierarchy.
    """

    def __init__(self):
        self.taxonomy = TAXONOMY

    def predict(self, customer_message: str, conversation_context: str = "") -> str:
        """
        Predict the customer's primary intent from the message and preceding context.
        """
        msg = clean_text(customer_message)
        ctx = clean_text(conversation_context)

        # -------------------------------------------------------------
        # 1. PURE RESOLUTION / COURTESY / THANK YOU -> GENERAL_INQUIRY_AND_OTHER
        # Precedence: Pure polite closure overrides prior topic in context.
        # -------------------------------------------------------------
        thank_you_pattern = (
            r'^(thanks?|thank you|thx|thks|appreciate it|much appreciated|awesome thank you|'
            r'great thanks|ok thanks|perfect thanks|thanks so much|thank you so much|'
            r'thanks for the help|will do thanks|good day)[.!]?$'
        )
        if re.search(thank_you_pattern, msg) or (
            len(msg.split()) <= 6
            and any(w in msg for w in ['thank you', 'thanks', 'thx', 'appreciate'])
            and not any(neg in msg for neg in ['no thanks', 'not', 'but', "didn't", 'worst', 'bad', 'problem', 'delay', 'issue', 'agent', 'service', 'card'])
        ):
            if not any(k in msg for k in ['refund', 'cancel', 'delay', 'rebook', 'bag', 'seat', 'lost', 'where', 'why', 'how', 'dm sent', 'customer service']):
                return "GENERAL_INQUIRY_AND_OTHER"

        if re.search(r'\b(finally resolved|got resolved|all set now|resolved.*thanks)\b', msg):
            return "GENERAL_INQUIRY_AND_OTHER"

        # -------------------------------------------------------------
        # 2. PHYSICAL SEATS & CABIN OVERRIDE
        # Physical seat discomfort, exit row questions, broken seat.
        # -------------------------------------------------------------
        seat_explicit_patterns = [
            r'\b(exit seat|exit row)\b.*(bag|legroom|seat)',
            r'\b(can\'t fit in a seat|fit in a seat|uncomfortable hours)\b',
            r'\b(bumped me from a seat|bclass seats?|broken seat|seat.*broken)\b',
            r'\b(disappointed w/ur .* seats?|seats? on aa\d+)\b',
        ]
        if any(re.search(p, msg) for p in seat_explicit_patterns):
            return "SEATS_AND_CABIN"

        # -------------------------------------------------------------
        # 3. REFUNDS & PAYMENTS (Highest Actionable Financial Precedence)
        # refund / credit / compensation / fees / vouchers / charge disputes
        # -------------------------------------------------------------
        refund_action_patterns = [
            r'\b(refund|refunds|refunded|refunding)\b',
            r'\b(reimburse|reimbursement|reimbursed)\b',
            r'\b(compensate|compensation|made whole)\b',
            r'\b(voucher|vouchers)\b',
            r'\b(credit|credits)\b.*(for this|toward|back|or something)',
            r'\b(a credit|some kind of credit|flight credit)\b',
            r'\b(money back|credit back|pay me back|pay back)\b',
            r'\b(pay twice|charged twice|double charged|charged me twice|overcharged|overcharge)\b',
            r'\b(charge|charges|charged|upcharge)\b.*\b(\$\s*\d+|\d+\s*dollars?|extra|additional|unfair)',
            r'\b(\$\s*\d+)\b.*\b(upcharge|charge)\b',
            r'\b(pay (a |the )?fee|pay for the change|change fee|baggage fee|minor fees?|cancellation fee)\b',
            r'\b(why did you charge|why was i charged|waive the fee|fee waiver|cancel fee)\b',
            r'\b(pay for expenses|cover expenses|cover my hotel|pay for my hotel|won\'t pay for expenses)\b',
            r'\b(credit card|charge|receipt|billing)\b.*(refund|dispute|incorrect|wrong|double)',
        ]
        has_refund_claim = any(re.search(p, msg) for p in refund_action_patterns)
        # Exclude explicit negations like "w/o refund" or venting "charge me all the fees you want"
        if has_refund_claim and not re.search(r'\b(w/o refund|without refund|charge me all the fees you want)\b', msg):
            if not re.search(r'\b(how (do|can) i pay|pay with miles)\b', msg):
                return "REFUNDS_AND_PAYMENTS"

        # Contextual refund request follow-up
        if any(re.search(p, ctx) for p in [r'\b(cancel.*flight.*insurance|refund.*ticket)\b']) and any(w in msg for w in ['forget it', 'booked thru', 'insurance']):
            return "REFUNDS_AND_PAYMENTS"

        # -------------------------------------------------------------
        # 4. BAGGAGE ISSUES & LOST ITEMS (Lost luggage, items left on plane)
        # -------------------------------------------------------------
        lost_item_patterns = [
            r'\b(left my (wallet|hat|phone|laptop|bag|item|passport|jacket))\b',
            r'\b(lost (and found|my (wallet|hat|phone|laptop|item)))\b',
            r'\b(get it back|scared it\'s halfway)\b.*(seat|flight|plane)',
        ]
        if any(re.search(p, msg) for p in lost_item_patterns):
            return "BAGGAGE_ISSUES"

        baggage_core_patterns = [
            r'\b(lost (bag|luggage|suitcase)|missing (bag|luggage|suitcase)|delayed (bag|luggage))\b',
            r'\b(luggage (never|didn\'t|did not) (make it|arrive)|no luggage|where is my (bag|luggage))\b',
            r'\b(baggage claim|bag delivery|track(ing)? (of )?(your |my )?luggage)\b',
            r'\b(damaged (bag|luggage|suitcase)|broken (wheel|handle|zipper))\b',
        ]
        if any(re.search(p, msg) for p in baggage_core_patterns):
            return "BAGGAGE_ISSUES"

        # Context baggage signals
        if re.search(r'\b(dm sent|i made it)\b', msg) and any(k in ctx for k in ['receipt for my bag', 'baggage', 'boarding my dog']):
            return "BAGGAGE_ISSUES"

        # -------------------------------------------------------------
        # 5. REBOOKING & FLIGHT CHANGES
        # -------------------------------------------------------------
        rebooking_patterns = [
            r'\b(rebook|rebooked|rebooking)\b',
            r'\b(change my flight|change flight|change (our|the) flight|change reservation|change dates)\b',
            r'\b(switch (to|flights?|my flight)|switch me to)\b',
            r'\b(standby|stand by|stand-by)\b',
            r'\b(put (me|us) on (the|a|another|next) flight)\b',
            r'\b(next available flight|earlier flight|later flight|different flight|alternate flight)\b',
            r'\b(get (me|us) on a flight|reschedule)\b',
            r'\b(booked on a connecting flight which arrive)\b',
            r'\b(missed (my|our|connection)|miss my connection)\b.*(options?|get me|rebook|what can)',
            r'\b(cancel a flight w/o refund)\b',
        ]
        if any(re.search(p, msg) for p in rebooking_patterns):
            return "REBOOKING_AND_CHANGES"

        if re.search(r'\b(dm sent|dm\'d|sent dm)\b', msg) and any(k in ctx for k in ['help rebooking', 'rebook', 'next available option', 'alternate flight']):
            return "REBOOKING_AND_CHANGES"

        # -------------------------------------------------------------
        # 6. FLIGHT DISRUPTION (Delays, cancellations, tarmac holds, mechanical)
        # -------------------------------------------------------------
        disruption_patterns = [
            r'\b(delay|delayed|delays|aadelays)\b',
            r'\b(cancel|cancelled|canceled|cancellation|cancellations)\b',
            r'\b(divert|diverted|diversion)\b',
            r'\b(tarmac|stuck on the plane|sitting on the tarmac|ground hold|runway for)\b',
            r'\b(mechanical|maintenance|crew timed out|maintenance delay|deplane)\b',
            r'\b(flight status|why is (the|my) flight|what is happening with flight)\b',
            r'\b(depart at .* yet to depart|hours late|hours delayed|departure now|was supposed to depart)\b',
            r'\b(was supposed to be in .* ago)\b',
            r'\b(missed connection|missed my connection)\b',
            r'\b(stranded.*customers|stranded)\b',
        ]
        if any(re.search(p, msg) for p in disruption_patterns):
            if not re.search(r'\b(change my|switch|standby|refund)\b', msg):
                return "FLIGHT_DISRUPTION"

        # -------------------------------------------------------------
        # 7. CUSTOMER SERVICE COMPLAINTS & EMPLOYEE BEHAVIOR
        # -------------------------------------------------------------
        cs_patterns = [
            r'\b(customer service|customer support)\b',
            r'\b(on hold|been on hold|holding for)\b',
            r'\b(representative|agent|employee|staff|crew|attendant|supervisor|manager)\b.*(rude|inept|useless|unhelpful|incompetent|terrible|awful|ignored|lied|yelling|screwing|zero help|quizzed|kudos|helpful|class|professionalism|said|watch your language)',
            r'\b(your employee said)\b',
            r'\b(attendant|ticket agent|gate agent)\b.*(handled|kudos|professional|impressed|shame|loud|yelling)',
            r'\b(flight attendant)\b',
            r'\b(rude|disrespectful|unprofessional|condescending|nasty|attitude|yelled at|arguing with)\b',
            r'\b(shame on you|disgrace|boycott|never fly(ing)? (again|with aa|with american)|do not fly aa|booking with (delta|united)|fly delta)\b',
            r'\b(no one (is )?(helping|cares|answers|answering|responding)|nobody (is )?(helping|cares|answers|responding))\b',
            r'\b(how long does it take to get a response|no response from|lack of follow up|done all (u|you) asked and no one)\b',
            r'\b(super dirty|filthy|disgusting|dirty sheets|moldy)\b',
            r'\b(unbelievably bad|pathetic|zero customer service|worst airline)\b',
            r'\b(you have my record locator|shouldn\'t have to do extra work|why do you need my record locator)\b',
            r'\b(disgusted|unacceptable|absolute joke|terrible experience|ruined my trip|ruined our vacation)\b',
            r'\b(poorservice|neveragain)\b',
            r'\b(anger and disgust at your service|wrote you on 2 outstanding employees)\b',
            r'\b(impressed with the customer service|excellent customer service)\b',
            r'\b(screwing me over|thnx for nothing|thanks for nothing)\b',
            r'\b(zero help whatsoever|nothing they can do)\b',
        ]
        if any(re.search(p, msg) for p in cs_patterns):
            return "CUSTOMER_SERVICE_COMPLAINT"

        # -------------------------------------------------------------
        # 8. CHECK-IN & BOARDING
        # -------------------------------------------------------------
        checkin_boarding_patterns = [
            r'\b(check in|check-in|checking in|checked in)\b',
            r'\b(boarding pass|boarded|boarding|board the plane|about to board)\b',
            r'\b(boarding group|group \d+|priority boarding|early boarding)\b',
            r'\b(gate agent|at the gate|gate \w+\d+|allowed to board|denied boarding|step out of line)\b',
            r'\b(carry on|carry-on|personal item|bag policy|size limit|overhead bin)\b.*(board|allowed|gate|group|allow)',
            r'\b(how many carry on|carry on allowed|overhead bins)\b',
            r'\b(tsa|security checkpoint|precheck|clear)\b',
            r'\b(online check-in|app check-in|check in times|earliest passengers may check)\b',
        ]
        if any(re.search(p, msg) for p in checkin_boarding_patterns):
            return "CHECKIN_AND_BOARDING"

        # -------------------------------------------------------------
        # 9. GENERAL BAGGAGE ISSUES
        # -------------------------------------------------------------
        general_baggage = [
            r'\b(bag|bags|baggage|luggage|suitcase|suitcases)\b',
            r'\b(checked (bag|bags|luggage)|check (a |my )?bag)\b',
        ]
        if any(re.search(p, msg) for p in general_baggage):
            return "BAGGAGE_ISSUES"

        # -------------------------------------------------------------
        # 10. SEATS & CABIN
        # -------------------------------------------------------------
        seat_cabin_patterns = [
            r'\b(seat|seats|seating|assigned seat|seat assignment)\b',
            r'\b(exit row|exit seat|middle seat|window seat|aisle seat|bulkhead|extra legroom|main cabin extra|first class seat)\b',
            r'\b(upgrade to first|first class upgrade|business class seat)\b',
            r'\b(wifi|wi-fi|internet on board|in-flight entertainment|ife|screen|power outlet|charging port)\b',
            r'\b(leg room|legroom|seat pitch|broken seat|recline)\b',
            r'\b(under (my|the) seat|fit in a seat)\b',
        ]
        if any(re.search(p, msg) for p in seat_cabin_patterns):
            return "SEATS_AND_CABIN"

        # -------------------------------------------------------------
        # 11. LOYALTY & AADVANTAGE
        # -------------------------------------------------------------
        is_general_praise = any(p in msg for p in [
            'what a beautiful fleet', 'looking forward to planning another trip',
            'blogging about our shared', 'taking care of travelers 24/7'
        ])
        if not is_general_praise:
            loyalty_core_patterns = [
                r'\b(aadvantage|aadv)\b',
                r'\b(miles that i earned|find our miles|bonus miles|earned miles|award travel|award ticket|award flight)\b',
                r'\b(admiral\'?s club|admirals club)\b',
                r'\b(concierge\s*key|#conciergekey)\b',
                r'\b(executive\s*platinum|#executiveplatinum)\b',
                r'\b(match the status|status match|former elites|elite status)\b',
                r'\b(gold aadvantage status|gold status|platinum card)\b',
                r'\b(loyalty doesn\'t matter|loyalty with #delta|what\'s the point of being loyal)\b',
                r'\b(systemwide upgrade|swu|mileage upgrade)\b',
                r'\b(frequent flyer|frequentflyer|million miler)\b',
                r'\b(miles|mileage)\b',
                r'\b(status|gold|platinum)\b',
            ]
            if any(re.search(p, msg) for p in loyalty_core_patterns):
                return "LOYALTY_AND_AADVANTAGE"

        # -------------------------------------------------------------
        # 12. BOOKING & RESERVATIONS
        # -------------------------------------------------------------
        booking_patterns = [
            r'\b(book a flight|booking a flight|want to book|trying to book|help me book)\b',
            r'\b(confirm a booking|confirmation number)\b',
            r'\b(fare|ticket price|buy a ticket|purchase ticket)\b',
            r'\b(name on ticket|spelling on ticket|itinerary)\b',
            r'\b(book|booking|booked|reserve|reservation)\b',
        ]
        if any(re.search(p, msg) for p in booking_patterns):
            if any(neg in msg for neg in ['disgusted', 'terrible', 'horrible', 'avoid', 'cancel', 'delay', 'ruined']):
                return "CUSTOMER_SERVICE_COMPLAINT"
            return "BOOKING_AND_RESERVATIONS"

        # Remaining General Complaints
        general_complaint_patterns = [
            r'\b(unacceptable|ridiculous|disgusted|furious|appalled|pissed|joke|scam|liars)\b',
            r'\b(sucks|terrible|horrible|awful|pathetic|fail|disaster|ruined)\b',
            r'\b(avoid @americanair|boycott|never again)\b',
            r'\b(discriminate|discrimination)\b',
        ]
        if any(re.search(p, msg) for p in general_complaint_patterns):
            return "CUSTOMER_SERVICE_COMPLAINT"

        # Contextual Fallbacks
        if any(k in ctx for k in ['delay', 'cancelled', 'weather', 'maintenance']):
            if any(k in msg for k in ['why', 'update', 'status', 'how long', 'take off', 'need to make that trip']):
                return "FLIGHT_DISRUPTION"
        if any(k in ctx for k in ['baggage', 'luggage']):
            return "BAGGAGE_ISSUES"

        return "GENERAL_INQUIRY_AND_OTHER"
