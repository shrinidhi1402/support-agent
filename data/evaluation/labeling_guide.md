# AmericanAir Customer Support: 10-Intent Labeling Guide

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
