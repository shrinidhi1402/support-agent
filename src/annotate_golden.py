"""
annotate_golden.py - Fast Batch Annotation Tool for AmericanAir Golden Set (10 Examples Per Page)

Usage:
    python src/annotate_golden.py
    python src/annotate_golden.py --port 5000
    python src/annotate_golden.py --no-browser
"""

import os
import sys
import argparse
import threading
import socket
import webbrowser
import pandas as pd
from flask import Flask, jsonify, request, render_template_string

# -----------------------------------------------------------------------------
# Configuration & Taxonomy
# -----------------------------------------------------------------------------
CSV_PATH = os.path.join("data", "evaluation", "golden_set.csv")

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

TAXONOMY_DESCRIPTIONS = {
    "FLIGHT_DISRUPTION": "Delays, cancellations, diversions, tarmac holds, mechanical/weather disruptions.",
    "REBOOKING_AND_CHANGES": "Changing dates/flights, standby, missing connections, alternative itineraries.",
    "BAGGAGE_ISSUES": "Lost, delayed, damaged, or mishandled bags, baggage fees, carry-on allowances.",
    "SEATS_AND_CABIN": "Seat selection, comfort, legroom, cabin cleanliness, in-flight WiFi/amenities.",
    "BOOKING_AND_RESERVATIONS": "New flight bookings, schedule inquiries, name corrections, record locators.",
    "REFUNDS_AND_PAYMENTS": "Refund requests, voucher compensation, charge disputes, pricing discrepancies.",
    "CHECKIN_AND_BOARDING": "Airport/app check-in issues, boarding pass, boarding groups, gate procedures.",
    "LOYALTY_AND_AADVANTAGE": "AAdvantage miles, tier status, mileage redemption, partner airlines.",
    "CUSTOMER_SERVICE_COMPLAINT": "Staff attitude, phone wait times, lack of assistance, generalized brand frustration.",
    "GENERAL_INQUIRY_AND_OTHER": "General policies, routine questions, positive feedback, greetings, miscellaneous.",
}

# -----------------------------------------------------------------------------
# Data Store Helpers
# -----------------------------------------------------------------------------
lock = threading.Lock()

def load_data() -> pd.DataFrame:
    """Load golden set preserving all columns and types."""
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Dataset not found at {CSV_PATH}. Run src/create_golden_set.py first.")
    
    df = pd.read_csv(
        CSV_PATH,
        dtype={
            "example_id": int,
            "conversation_id": int,
            "tweet_id": int,
            "customer_message": str,
            "conversation_context": str,
            "candidate_intent": str,
            "final_intent": str,
            "difficulty": str,
            "notes": str,
        },
        keep_default_na=False,
    )
    return df

def save_data(df: pd.DataFrame):
    """Safely persist dataset to CSV atomically."""
    with lock:
        tmp_path = CSV_PATH + ".tmp"
        df.to_csv(tmp_path, index=False, encoding="utf-8")
        if os.path.exists(CSV_PATH):
            os.replace(tmp_path, CSV_PATH)
        else:
            os.rename(tmp_path, CSV_PATH)

# Initialize dataset
df_dataset = load_data()

# -----------------------------------------------------------------------------
# Web UI Template (Single-File Batch Annotation)
# -----------------------------------------------------------------------------
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AmericanAir Golden Set: Fast Batch Annotation (10/page)</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0d1117;
      --card-bg: #161b22;
      --card-hover: #1c2128;
      --border: #30363d;
      --text: #c9d1d9;
      --text-bright: #f0f6fc;
      --muted: #8b949e;
      --accent: #58a6ff;
      --accent-hover: #1f6feb;
      --success: #238636;
      --success-light: #2ea043;
      --warning: #d29922;
      --danger: #f85149;
      --purple: #a371f7;
      --candidate-bg: #1f242c;
      --radius: 8px;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding-bottom: 60px;
    }

    /* Top Sticky Navigation Bar */
    header {
      background-color: var(--card-bg);
      border-bottom: 1px solid var(--border);
      padding: 12px 24px;
      position: sticky;
      top: 0;
      z-index: 200;
      box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .header-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 16px;
      margin-bottom: 10px;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .brand h1 {
      font-size: 1.1rem;
      font-weight: 700;
      color: var(--text-bright);
    }
    .brand span {
      background: #21262d;
      color: var(--accent);
      padding: 2px 8px;
      border-radius: 12px;
      font-size: 0.75rem;
      font-weight: 600;
      border: 1px solid var(--border);
    }

    .progress-section {
      display: flex;
      align-items: center;
      gap: 16px;
    }
    .stat-pill {
      font-size: 0.82rem;
      background: #21262d;
      border: 1px solid var(--border);
      padding: 4px 10px;
      border-radius: 6px;
      color: var(--text-bright);
    }
    .stat-pill strong {
      color: #3fb950;
    }
    .stat-pill.remaining strong {
      color: #e3b341;
    }
    .progress-bar-container {
      width: 140px;
      height: 8px;
      background: #21262d;
      border-radius: 4px;
      overflow: hidden;
      border: 1px solid var(--border);
    }
    .progress-bar-fill {
      height: 100%;
      background: var(--success-light);
      width: 0%;
      transition: width 0.3s ease;
    }

    /* Filter & Pagination Controls */
    .controls-bar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
      border-top: 1px solid #21262d;
      padding-top: 10px;
    }
    .filter-group {
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .filter-label {
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: var(--muted);
      font-weight: 600;
      margin-right: 4px;
    }
    .filter-btn {
      background: #21262d;
      border: 1px solid var(--border);
      color: var(--text);
      padding: 4px 12px;
      border-radius: 16px;
      cursor: pointer;
      font-size: 0.8rem;
      font-weight: 500;
      transition: all 0.15s ease;
    }
    .filter-btn:hover {
      background: #30363d;
      color: var(--text-bright);
    }
    .filter-btn.active {
      background: var(--accent);
      color: #ffffff;
      border-color: var(--accent);
      font-weight: 600;
    }

    .pagination-group {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .page-info {
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--text-bright);
      padding: 0 4px;
    }
    .btn {
      background: #21262d;
      border: 1px solid var(--border);
      color: var(--text);
      padding: 6px 14px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 0.82rem;
      font-weight: 600;
      transition: all 0.15s ease;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .btn:hover:not(:disabled) {
      background: #30363d;
      color: var(--text-bright);
      border-color: #8b949e;
    }
    .btn:disabled {
      opacity: 0.4;
      cursor: not-allowed;
    }
    .btn-primary {
      background: var(--accent);
      color: #ffffff;
      border: none;
    }
    .btn-primary:hover:not(:disabled) {
      background: var(--accent-hover);
    }
    .btn-accept {
      background: #238636;
      color: #ffffff;
      border: 1px solid #2ea043;
    }
    .btn-accept:hover {
      background: #2ea043;
    }
    .btn-skip {
      background: #21262d;
      color: var(--muted);
      border: 1px solid var(--border);
    }
    .btn-skip:hover {
      background: #30363d;
      color: var(--text);
    }
    .btn-clear {
      background: transparent;
      color: var(--danger);
      border: 1px solid rgba(248, 81, 73, 0.3);
      padding: 4px 8px;
      font-size: 0.75rem;
    }
    .btn-clear:hover {
      background: rgba(248, 81, 73, 0.15);
    }

    /* Main Grid Layout */
    main {
      max-width: 1280px;
      margin: 20px auto;
      padding: 0 20px;
      display: grid;
      grid-template-columns: 1fr 310px;
      gap: 20px;
    }

    /* Examples Stream (10 per page) */
    .examples-stream {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .example-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 16px 18px;
      display: flex;
      flex-direction: column;
      gap: 12px;
      transition: border-color 0.2s ease, background 0.2s ease;
      position: relative;
    }
    .example-card.reviewed {
      border-color: rgba(46, 160, 67, 0.6);
      background: #141a17;
    }
    .example-card.skipped {
      opacity: 0.75;
    }

    .card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 8px;
      border-bottom: 1px solid #21262d;
      padding-bottom: 8px;
    }
    .meta-tags {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    .meta-badge {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.75rem;
      background: #21262d;
      color: var(--text-bright);
      padding: 2px 8px;
      border-radius: 4px;
      border: 1px solid var(--border);
    }
    .badge-diff {
      font-size: 0.72rem;
      font-weight: 700;
      padding: 2px 8px;
      border-radius: 12px;
      text-transform: uppercase;
    }
    .diff-easy { background: rgba(46, 160, 67, 0.15); color: #3fb950; border: 1px solid rgba(46, 160, 67, 0.3); }
    .diff-medium { background: rgba(210, 153, 34, 0.15); color: #e3b341; border: 1px solid rgba(210, 153, 34, 0.3); }
    .diff-hard { background: rgba(248, 81, 73, 0.15); color: #f85149; border: 1px solid rgba(248, 81, 73, 0.3); }

    .review-status-badge {
      font-size: 0.75rem;
      font-weight: 700;
      padding: 3px 10px;
      border-radius: 12px;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-family: 'JetBrains Mono', monospace;
    }
    .status-unreviewed {
      background: rgba(210, 153, 34, 0.12);
      color: #e3b341;
      border: 1px solid rgba(210, 153, 34, 0.3);
    }
    .status-reviewed {
      background: rgba(46, 160, 67, 0.2);
      color: #3fb950;
      border: 1px solid rgba(46, 160, 67, 0.4);
    }

    /* Context Section */
    .context-block {
      background: #0d1117;
      border: 1px solid #21262d;
      border-radius: 6px;
      padding: 8px 12px;
      font-size: 0.8rem;
      color: #8b949e;
      line-height: 1.5;
      max-height: 120px;
      overflow-y: auto;
    }
    .context-block .turn {
      margin-bottom: 4px;
    }
    .context-block .turn:last-child {
      margin-bottom: 0;
    }
    .turn-cust { color: #58a6ff; font-weight: 600; }
    .turn-air { color: #f0883e; font-weight: 600; }

    /* Customer Message Section */
    .customer-msg-box {
      background: #1c2128;
      border-left: 3px solid var(--accent);
      border-radius: 4px;
      padding: 12px 14px;
    }
    .customer-msg-text {
      font-size: 0.98rem;
      font-weight: 500;
      color: var(--text-bright);
      line-height: 1.55;
      word-break: break-word;
    }

    /* Sampling Notes */
    .notes-snippet {
      font-size: 0.78rem;
      color: #d2a8ff;
      background: rgba(163, 113, 247, 0.08);
      border-radius: 4px;
      padding: 4px 10px;
      border-left: 2px solid var(--purple);
    }

    /* Actions Row */
    .card-actions {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
      background: #11151c;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 10px 14px;
    }
    .candidate-display {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    .cand-label {
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: var(--muted);
      font-weight: 600;
    }
    .cand-intent-val {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.85rem;
      color: #e3b341;
      font-weight: 600;
      background: #1f242c;
      padding: 2px 8px;
      border-radius: 4px;
      border: 1px solid #30363d;
    }

    .action-controls {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    .intent-select {
      background: #0d1117;
      border: 1px solid var(--border);
      color: var(--text-bright);
      padding: 6px 10px;
      border-radius: 6px;
      font-size: 0.82rem;
      font-weight: 500;
      cursor: pointer;
      outline: none;
    }
    .intent-select:focus {
      border-color: var(--accent);
    }

    /* Right Sidebar: Distribution Summary */
    .sidebar {
      display: flex;
      flex-direction: column;
      gap: 16px;
      position: sticky;
      top: 130px;
      height: fit-content;
    }
    .side-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 16px;
    }
    .side-card h3 {
      font-size: 0.88rem;
      font-weight: 700;
      color: var(--text-bright);
      margin-bottom: 12px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .intent-counts-list {
      display: flex;
      flex-direction: column;
      gap: 6px;
      font-size: 0.78rem;
    }
    .count-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 4px 6px;
      border-radius: 4px;
      background: #0d1117;
    }
    .count-row .intent-name {
      font-family: 'JetBrains Mono', monospace;
      color: var(--text);
      font-size: 0.75rem;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      max-width: 190px;
    }
    .count-row .intent-badge {
      font-family: 'JetBrains Mono', monospace;
      font-weight: 700;
      padding: 1px 6px;
      border-radius: 10px;
      font-size: 0.72rem;
    }
    .badge-has-count {
      background: rgba(46, 160, 67, 0.2);
      color: #3fb950;
      border: 1px solid rgba(46, 160, 67, 0.4);
    }
    .badge-zero {
      background: #21262d;
      color: var(--muted);
    }

    /* Page Navigation Bottom Bar */
    .bottom-nav {
      max-width: 1280px;
      margin: 20px auto 0;
      padding: 0 20px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
    }

    /* Empty state */
    .empty-state {
      background: var(--card-bg);
      border: 1px dashed var(--border);
      border-radius: var(--radius);
      padding: 40px;
      text-align: center;
      color: var(--muted);
    }
    .empty-state h3 {
      color: var(--text-bright);
      margin-bottom: 8px;
    }

    /* Toast Notification */
    #toast {
      position: fixed;
      bottom: 20px;
      right: 20px;
      background: #238636;
      color: #fff;
      padding: 8px 16px;
      border-radius: 6px;
      font-size: 0.85rem;
      font-weight: 600;
      opacity: 0;
      transition: opacity 0.3s ease;
      z-index: 1000;
      box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    }
    #toast.show {
      opacity: 1;
    }

    /* Responsive */
    @media (max-width: 960px) {
      main { grid-template-columns: 1fr; }
      .sidebar { position: static; }
    }
  </style>
</head>
<body>

<header>
  <div class="header-top">
    <div class="brand">
      <h1>AmericanAir Golden Set Annotation</h1>
      <span>Batch Mode (10 / page)</span>
    </div>

    <div class="progress-section">
      <div class="stat-pill">
        Reviewed: <strong id="stat-reviewed">0</strong> / 200 (<span id="stat-pct">0%</span>)
      </div>
      <div class="stat-pill remaining">
        Remaining: <strong id="stat-remaining">200</strong>
      </div>
      <div class="progress-bar-container">
        <div class="progress-bar-fill" id="stat-bar"></div>
      </div>
    </div>
  </div>

  <div class="controls-bar">
    <div class="filter-group">
      <span class="filter-label">Filter:</span>
      <button class="filter-btn active" data-filter="unlabelled">Unlabelled</button>
      <button class="filter-btn" data-filter="all">All</button>
      <button class="filter-btn" data-filter="hard">Hard</button>
      <button class="filter-btn" data-filter="medium">Medium</button>
      <button class="filter-btn" data-filter="easy">Easy</button>
    </div>

    <div class="pagination-group">
      <button class="btn" id="btn-top-prev" title="Previous Page">&larr; Prev</button>
      <span class="page-info" id="top-page-info">Page 1 of 1</span>
      <button class="btn" id="btn-top-next" title="Next Page">Next &rarr;</button>
      <button class="btn btn-primary" id="btn-save-next" style="margin-left: 8px;">Save & Next Page &rarr;</button>
    </div>
  </div>
</header>

<main>
  <!-- 10 Examples List -->
  <section class="examples-stream" id="examples-container">
    <div class="empty-state">
      <h3>Loading batch examples...</h3>
    </div>
  </section>

  <!-- Sidebar: Live Intent Counts Breakdown -->
  <aside class="sidebar">
    <div class="side-card">
      <h3>
        <span>Final Intent Distribution</span>
        <span id="side-total-labelled" style="font-size: 0.75rem; color: #3fb950; font-weight: 600;">0 total</span>
      </h3>
      <div class="intent-counts-list" id="intent-distribution-list">
        <!-- Rendered via JS -->
      </div>
    </div>

    <div class="side-card" style="font-size: 0.78rem; color: var(--muted); line-height: 1.6;">
      <h3 style="font-size: 0.82rem;">Fast Batch Tips</h3>
      <p>&#8226; Click <strong>Accept Candidate</strong> if the heuristic is correct.</p>
      <p>&#8226; Use the <strong>Dropdown</strong> if another intent fits better.</p>
      <p>&#8226; Every action <strong>saves immediately</strong> to CSV.</p>
      <p>&#8226; Click <strong>Save & Next Page</strong> when done with the current batch.</p>
    </div>
  </aside>
</main>

<div class="bottom-nav">
  <div style="font-size: 0.82rem; color: var(--muted);">
    Showing up to 10 examples per page. Progress is continuously preserved on disk.
  </div>
  <div class="pagination-group">
    <button class="btn" id="btn-bot-prev">&larr; Prev Page</button>
    <span class="page-info" id="bot-page-info">Page 1 of 1</span>
    <button class="btn" id="btn-bot-next">Next Page &rarr;</button>
  </div>
</div>

<div id="toast">Saved!</div>

<script>
  const TAXONOMY = {{ taxonomy | tojson }};
  let currentFilter = 'unlabelled';
  let currentPage = 1;
  let totalPages = 1;

  // Show Toast
  function showToast(msg = 'Saved!') {
    const toast = document.getElementById('toast');
    toast.textContent = msg;
    toast.className = 'show';
    setTimeout(() => { toast.className = ''; }, 1400);
  }

  // Load Page Data
  async function loadPage(page = 1) {
    try {
      const resp = await fetch(`/api/batch_page?filter=${currentFilter}&page=${page}&page_size=10`);
      if (!resp.ok) {
        alert("Failed to load page");
        return;
      }
      const data = await resp.json();
      currentPage = data.page;
      totalPages = data.total_pages;

      renderExamples(data.examples);
      updateHeaderPagination(data.page, data.total_pages, data.total_items);
      updateStats(data.stats);
    } catch (err) {
      console.error("Error loading page:", err);
    }
  }

  // Render Examples on Page
  function renderExamples(examples) {
    const container = document.getElementById('examples-container');
    container.innerHTML = '';

    if (!examples || examples.length === 0) {
      container.innerHTML = `
        <div class="empty-state">
          <h3>No examples found in "${currentFilter}" filter!</h3>
          <p style="margin-top: 6px;">All examples in this category have been processed or none match.</p>
        </div>
      `;
      return;
    }

    examples.forEach(ex => {
      const card = document.createElement('div');
      const isReviewed = ex.final_intent && ex.final_intent.trim() !== '';
      card.className = `example-card ${isReviewed ? 'reviewed' : ''}`;
      card.id = `card-${ex.example_id}`;

      // Format context turns
      let contextHtml = '';
      const rawContext = ex.conversation_context || '';
      if (!rawContext || rawContext.includes('[No preceding context')) {
        contextHtml = '<span style="color: var(--muted); font-style: italic;">[Conversation start / No preceding turns]</span>';
      } else {
        const turns = rawContext.split(/\s*\|\s*(?=\[\d+\])/g);
        turns.forEach(t => {
          let formatted = t;
          if (t.includes('Customer')) {
            formatted = `<span class="turn-cust">&#9656; Customer:</span> ` + t.replace(/^\[\d+\]\s*Customer\s*\(\d+\):\s*/i, '');
          } else if (t.includes('AmericanAir')) {
            formatted = `<span class="turn-air">&#9656; AmericanAir:</span> ` + t.replace(/^\[\d+\]\s*AmericanAir:\s*/i, '');
          }
          contextHtml += `<div class="turn">${formatted}</div>`;
        });
      }

      // Dropdown options
      let dropdownOptions = `<option value="">-- Choose different intent --</option>`;
      TAXONOMY.forEach(t => {
        const isSelected = ex.final_intent === t;
        dropdownOptions += `<option value="${t}" ${isSelected ? 'selected' : ''}>${t}</option>`;
      });

      card.innerHTML = `
        <div class="card-header">
          <div class="meta-tags">
            <span class="meta-badge">Ex #${ex.example_id}</span>
            <span class="meta-badge">Conv #${ex.conversation_id}</span>
            <span class="meta-badge">Tweet #${ex.tweet_id}</span>
            <span class="badge-diff diff-${ex.difficulty.toLowerCase()}">${ex.difficulty}</span>
          </div>

          <div class="review-status-badge ${isReviewed ? 'status-reviewed' : 'status-unreviewed'}" id="status-badge-${ex.example_id}">
            ${isReviewed ? `&#10003; REVIEWED: ${ex.final_intent}` : `&#9675; UNREVIEWED`}
          </div>
        </div>

        <div>
          <div style="font-size: 0.72rem; text-transform: uppercase; color: var(--muted); font-weight: 600; margin-bottom: 4px;">
            Preceding Conversation Context
          </div>
          <div class="context-block">${contextHtml}</div>
        </div>

        <div>
          <div style="font-size: 0.72rem; text-transform: uppercase; color: var(--muted); font-weight: 600; margin-bottom: 4px;">
            Customer Message
          </div>
          <div class="customer-msg-box">
            <div class="customer-msg-text">${escapeHtml(ex.customer_message)}</div>
          </div>
        </div>

        ${ex.notes ? `<div class="notes-snippet"><strong>Notes:</strong> ${escapeHtml(ex.notes)}</div>` : ''}

        <div class="card-actions">
          <div class="candidate-display">
            <span class="cand-label">Candidate:</span>
            <span class="cand-intent-val">${ex.candidate_intent}</span>
            <button class="btn btn-accept" onclick="acceptCandidate(${ex.example_id}, '${ex.candidate_intent}')">
              &#10003; Accept Candidate
            </button>
          </div>

          <div class="action-controls">
            <select class="intent-select" id="select-${ex.example_id}" onchange="changeIntent(${ex.example_id}, this.value)">
              ${dropdownOptions}
            </select>
            <button class="btn btn-skip" onclick="skipCard(${ex.example_id})">Skip</button>
            ${isReviewed ? `<button class="btn btn-clear" onclick="clearIntent(${ex.example_id})">Clear</button>` : ''}
          </div>
        </div>
      `;

      container.appendChild(card);
    });
  }

  // HTML Escaper
  function escapeHtml(str) {
    if (!str) return '';
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // Accept Candidate Intent
  async function acceptCandidate(exampleId, candidateIntent) {
    await saveIntent(exampleId, candidateIntent);
  }

  // Change Intent from Dropdown
  async function changeIntent(exampleId, selectedIntent) {
    if (!selectedIntent) return;
    await saveIntent(exampleId, selectedIntent);
  }

  // Clear Intent back to blank
  async function clearIntent(exampleId) {
    await saveIntent(exampleId, "");
  }

  // Skip Card (Smooth Scroll to Next Card)
  function skipCard(exampleId) {
    const card = document.getElementById(`card-${exampleId}`);
    if (card) {
      card.classList.add('skipped');
      const nextCard = card.nextElementSibling;
      if (nextCard) {
        nextCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    }
  }

  // Save Intent API call
  async function saveIntent(exampleId, finalIntent) {
    try {
      const resp = await fetch('/api/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ example_id: exampleId, final_intent: finalIntent }),
      });
      const res = await resp.json();
      if (res.status === 'ok') {
        showToast(finalIntent ? `Saved: ${finalIntent}` : 'Cleared');
        
        // Update Card UI
        const card = document.getElementById(`card-${exampleId}`);
        const statusBadge = document.getElementById(`status-badge-${exampleId}`);
        const selectBox = document.getElementById(`select-${exampleId}`);

        if (finalIntent) {
          card.classList.add('reviewed');
          statusBadge.className = 'review-status-badge status-reviewed';
          statusBadge.innerHTML = `&#10003; REVIEWED: ${finalIntent}`;
          if (selectBox) selectBox.value = finalIntent;
        } else {
          card.classList.remove('reviewed');
          statusBadge.className = 'review-status-badge status-unreviewed';
          statusBadge.innerHTML = `&#9675; UNREVIEWED`;
          if (selectBox) selectBox.value = '';
        }

        // Refresh global statistics
        refreshStats();
      }
    } catch (err) {
      alert("Error saving: " + err);
    }
  }

  // Refresh Stats
  async function refreshStats() {
    try {
      const resp = await fetch('/api/stats');
      const stats = await resp.json();
      updateStats(stats);
    } catch (e) {
      console.error(e);
    }
  }

  // Update Stats UI
  function updateStats(stats) {
    document.getElementById('stat-reviewed').textContent = stats.labelled;
    document.getElementById('stat-remaining').textContent = stats.unlabelled;
    const pct = ((stats.labelled / stats.total) * 100).toFixed(1);
    document.getElementById('stat-pct').textContent = `${pct}%`;
    document.getElementById('stat-bar').style.width = `${pct}%`;
    document.getElementById('side-total-labelled').textContent = `${stats.labelled} / ${stats.total}`;

    // Update Sidebar Distribution List
    const distContainer = document.getElementById('intent-distribution-list');
    distContainer.innerHTML = '';

    TAXONOMY.forEach(intent => {
      const count = (stats.intent_counts && stats.intent_counts[intent]) || 0;
      const row = document.createElement('div');
      row.className = 'count-row';
      row.innerHTML = `
        <span class="intent-name" title="${intent}">${intent}</span>
        <span class="intent-badge ${count > 0 ? 'badge-has-count' : 'badge-zero'}">${count}</span>
      `;
      distContainer.appendChild(row);
    });

    // Add unlabelled row
    const unRow = document.createElement('div');
    unRow.className = 'count-row';
    unRow.style.borderTop = '1px dashed #30363d';
    unRow.style.marginTop = '4px';
    unRow.style.paddingTop = '6px';
    unRow.innerHTML = `
      <span class="intent-name" style="color: #e3b341;">(Unlabelled)</span>
      <span class="intent-badge ${stats.unlabelled > 0 ? 'badge-has-count' : 'badge-zero'}" style="${stats.unlabelled > 0 ? 'color: #e3b341; border-color: rgba(210,153,34,0.4);' : ''}">
        ${stats.unlabelled}
      </span>
    `;
    distContainer.appendChild(unRow);
  }

  // Pagination Text & Button State
  function updateHeaderPagination(page, total, totalItems) {
    const text = `Page ${page} of ${total || 1} (${totalItems} items)`;
    document.getElementById('top-page-info').textContent = text;
    document.getElementById('bot-page-info').textContent = text;

    document.getElementById('btn-top-prev').disabled = (page <= 1);
    document.getElementById('btn-bot-prev').disabled = (page <= 1);
    document.getElementById('btn-top-next').disabled = (page >= total);
    document.getElementById('btn-bot-next').disabled = (page >= total);
  }

  // Filter Buttons
  document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentFilter = btn.getAttribute('data-filter');
      currentPage = 1;
      loadPage(1);
    });
  });

  // Prev / Next Listeners
  function prevPage() {
    if (currentPage > 1) {
      loadPage(currentPage - 1);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  }
  function nextPage() {
    if (currentPage < totalPages) {
      loadPage(currentPage + 1);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  }

  document.getElementById('btn-top-prev').addEventListener('click', prevPage);
  document.getElementById('btn-bot-prev').addEventListener('click', prevPage);
  document.getElementById('btn-top-next').addEventListener('click', nextPage);
  document.getElementById('btn-bot-next').addEventListener('click', nextPage);

  document.getElementById('btn-save-next').addEventListener('click', () => {
    showToast('Page saved!');
    if (currentPage < totalPages) {
      nextPage();
    } else {
      // Reload current filter to pick up newly unlabelled or remaining
      loadPage(currentPage);
    }
  });

  // Initial Load (Defaults to Unlabelled page 1)
  loadPage(1);
</script>

</body>
</html>
"""

# -----------------------------------------------------------------------------
# Flask Application & Endpoints
# -----------------------------------------------------------------------------
app = Flask(__name__)

@app.route("/")
def index():
    return render_template_string(
        HTML_TEMPLATE,
        taxonomy=TAXONOMY,
        descriptions=TAXONOMY_DESCRIPTIONS,
    )

@app.route("/api/batch_page")
def get_batch_page():
    global df_dataset
    filter_type = request.args.get("filter", "unlabelled").lower().strip()
    page = int(request.args.get("page", 1))
    page_size = int(request.args.get("page_size", 10))

    # Apply Filter
    if filter_type == "unlabelled":
        filtered_df = df_dataset[df_dataset["final_intent"].str.strip() == ""]
    elif filter_type == "hard":
        filtered_df = df_dataset[df_dataset["difficulty"].str.lower() == "hard"]
    elif filter_type == "medium":
        filtered_df = df_dataset[df_dataset["difficulty"].str.lower() == "medium"]
    elif filter_type == "easy":
        filtered_df = df_dataset[df_dataset["difficulty"].str.lower() == "easy"]
    else:  # "all"
        filtered_df = df_dataset

    total_items = len(filtered_df)
    total_pages = max(1, (total_items + page_size - 1) // page_size)
    page = max(1, min(page, total_pages))

    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_slice = filtered_df.iloc[start_idx:end_idx]

    examples = []
    for _, row in page_slice.iterrows():
        examples.append({
            "example_id": int(row["example_id"]),
            "conversation_id": int(row["conversation_id"]),
            "tweet_id": int(row["tweet_id"]),
            "customer_message": str(row["customer_message"]),
            "conversation_context": str(row["conversation_context"]),
            "candidate_intent": str(row["candidate_intent"]),
            "final_intent": str(row["final_intent"]),
            "difficulty": str(row["difficulty"]),
            "notes": str(row["notes"]),
        })

    # Global Stats
    total = len(df_dataset)
    labelled = int((df_dataset["final_intent"].str.strip() != "").sum())
    unlabelled = total - labelled
    intent_counts = {intent: int((df_dataset["final_intent"] == intent).sum()) for intent in TAXONOMY}

    return jsonify({
        "examples": examples,
        "page": page,
        "page_size": page_size,
        "total_items": total_items,
        "total_pages": total_pages,
        "stats": {
            "total": total,
            "labelled": labelled,
            "unlabelled": unlabelled,
            "intent_counts": intent_counts,
        }
    })

@app.route("/api/save", methods=["POST"])
def save_single_intent():
    global df_dataset
    data = request.get_json(force=True)
    example_id = int(data.get("example_id"))
    final_intent = str(data.get("final_intent", "")).strip()

    if final_intent and final_intent not in TAXONOMY:
        return jsonify({"error": f"Invalid intent: {final_intent}"}), 400

    idx_matches = df_dataset.index[df_dataset["example_id"] == example_id]
    if len(idx_matches) == 0:
        return jsonify({"error": "Example ID not found"}), 404

    df_dataset.loc[idx_matches[0], "final_intent"] = final_intent
    save_data(df_dataset)

    return jsonify({"status": "ok", "example_id": example_id, "final_intent": final_intent})

@app.route("/api/stats")
def get_stats():
    global df_dataset
    total = len(df_dataset)
    labelled = int((df_dataset["final_intent"].str.strip() != "").sum())
    unlabelled = total - labelled
    intent_counts = {intent: int((df_dataset["final_intent"] == intent).sum()) for intent in TAXONOMY}
    return jsonify({
        "total": total,
        "labelled": labelled,
        "unlabelled": unlabelled,
        "intent_counts": intent_counts,
    })

# -----------------------------------------------------------------------------
# Port & Server Runner
# -----------------------------------------------------------------------------
def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0

def find_available_port(start_port: int = 5000, max_attempts: int = 20) -> int:
    for p in range(start_port, start_port + max_attempts):
        if not is_port_in_use(p):
            return p
    return start_port

def open_browser(url: str):
    threading.Timer(1.2, lambda: webbrowser.open_new(url)).start()

def main():
    parser = argparse.ArgumentParser(description="Fast Batch Annotation Tool for AmericanAir Golden Set")
    parser.add_argument("--port", type=int, default=5000, help="Port to run local UI (default: 5000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    args = parser.parse_args()

    # Verify dataset exists
    if not os.path.exists(CSV_PATH):
        print(f"Error: Dataset not found at '{CSV_PATH}'. Run src/create_golden_set.py first.")
        sys.exit(1)

    # Check port availability
    target_port = args.port
    if is_port_in_use(target_port):
        alt_port = find_available_port(target_port + 1)
        print(f"Notice: Port {target_port} is currently in use.")
        print(f"Switching automatically to available port: {alt_port}")
        target_port = alt_port

    url = f"http://127.0.0.1:{target_port}"
    print("=" * 68)
    print("AMERICAN AIRLINES GOLDEN SET - FAST BATCH ANNOTATION TOOL (10/PAGE)")
    print("=" * 68)
    print(f"Dataset path:    {CSV_PATH} ({len(df_dataset)} examples)")
    labelled_count = (df_dataset["final_intent"].str.strip() != "").sum()
    print(f"Current Status:  {labelled_count} reviewed / {len(df_dataset) - labelled_count} remaining")
    print(f"Local Server:    {url}")
    print("=" * 68)
    print("Features:")
    print("  - 10 examples per page with customer message, context, and candidate intent")
    print("  - 'Accept Candidate' button copies candidate to final_intent & saves immediately")
    print("  - Dropdown selector allows choosing any of the 10 intents")
    print("  - Filters: Unlabelled (Default), All, Hard, Medium, Easy")
    print("  - Live intent distribution counters and progress tracking")
    print("=" * 68)
    print("Press Ctrl+C in terminal to stop the server.")

    if not args.no_browser:
        print(f"Opening browser at {url} ...")
        open_browser(url)

    # Run local server
    app.run(host="127.0.0.1", port=target_port, debug=False)

if __name__ == "__main__":
    main()
