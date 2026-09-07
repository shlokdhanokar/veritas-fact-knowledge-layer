# Veritas — A Fact Knowledge Layer

> Superjoin VIT 2026 · Engineering Intern Assignment
> **Deadline: 2026-09-09, 11:00 AM IST**

---

## 1. The Problem

Build a system that ingests arbitrary PDFs and:

1. **extracts** meaningful numerical or semantic facts,
2. **grounds** every fact to exact evidence in its source document,
3. **relates** facts across documents — corroboration, contradiction, or reconciliation-by-context.

Exposed through an API/UI that accepts new PDFs. **No hardcoded facts, filenames, schemas, or
document-specific rules** — graders will test with their own PDFs.

### Explicit anti-goals (stated by Superjoin)
- "A graph database or visualization alone is not the solution."
- Production polish is *not* wanted. A small, understandable prototype beats a large opaque one.
- Perfect extraction is *not* expected. Honest failure analysis is explicitly requested.

---

## 2. The Core Idea — Context Keys

Most submissions will model a fact as `(entity, metric, value)`. That model *cannot* express the
difference between a real contradiction and a scope mismatch, so it produces false positives.

**A number is meaningless without the context that qualifies it.** Every fact is stored as:

```
subject · metric · value · unit
+ CONTEXT KEYS: period, basis, entity_scope, vintage, geography, segment
+ EVIDENCE:     doc_id, page, char_span, bbox, verbatim quote
+ PROVENANCE:   publisher, publication_date, estimate_stage
```

The comparison engine then follows **one rule**, and that rule generates all three required cases:

| Context keys | Values | Verdict |
|---|---|---|
| match | agree | **CORROBORATES** |
| match | disagree | **CONTRADICTS** |
| differ | either | **RECONCILED** — and we name the key that differs |

One mechanism, three outcomes. This is the thesis of the submission.

---

## 3. Verified Target Cases

Pre-verified by grepping all 1.8M characters of the six starter PDFs. These are real, not aspirational.

### Case 1 — Corroboration across documents, expressed differently
- Annual Report FY24: consolidated revenue from operations **₹81,415.38 million**
- Q4 FY24 earnings deck: **₹8,142 Cr** FY24 revenue

Different unit (Mn vs Cr), different rounding, different wording. Normalizer proves
₹8,142 Cr = ₹81,420 Mn ≈ ₹81,415.38 Mn within tolerance.

### Case 2 — Apparent contradiction resolved by ENTITY SCOPE (showpiece)
- Annual Report FY24, same table: revenue is **₹74,540.82M standalone** *and* **₹81,415.38M consolidated**
- Prospectus 2022, same page-region, same date: "**82 gateways** (excluding Spoton) as of December 31, 2021"
  vs "**122 gateways** ... as of December 31, 2021, including Spoton's **40 gateways**"

The gateway pair is **arithmetically provable**: 82 + 40 = 122. A reconciliation the system can *verify*,
not merely assert.

### Case 3 — Apparent contradiction resolved by DATA VINTAGE
India FY25 real GDP growth:
- Economic Survey 2024-25 → **6.4%** (First Advance Estimate, Jan 2025)
- RBI Annual Report 2024-25 → **6.5%** ("moderated to 6.5 per cent in 2024-25")
- IMF Article IV → **6.5%** ("real GDP grew by 6.5 percent in FY2024/25")

Same metric, same period, same country. Differs by **estimate vintage**. Resolution orders by
publication date and recognises advance → provisional revision.

### Case 4 — Temporal state contradiction (genuine)
- Prospectus 2022: "Sandeep Kumar Barasia **is** an Executive Director and Chief Business Officer"
- Annual Report FY24: "ceased to be a Director with effect from **July 01, 2024**"

Handled with **valid-time intervals** (`as_of` / `until`), so both facts are true and coexist.
Same pattern available for Suvir Suren Sujan (2023-08-24) and Donald Colleran (2023-09-27).

### Case 5 — Documented extraction failure (required, honest)
The FY24 financials table linearises to a flat run:
`Revenue from Operations 74,540.82 66,586.61 81,415.38 72,253.01`
under a **two-level header** (Standalone/Consolidated × FY24/FY23). A naive extractor binds the wrong
column and silently reports standalone revenue as consolidated. Silent, plausible, and exactly the
failure the context-key model exists to catch. Documented with mitigation + residual risk.

---

## 4. Architecture

```
PDF
 └─> Layout-aware parse (PyMuPDF)        text + page + char offsets + bbox
     └─> Evidence units (para / table row)
         └─> LLM claim extraction        typed schema, structured output, batched
             └─> Normalization           units, currency, fiscal periods, aliases   [DETERMINISTIC]
                 └─> Metric canonicalization                    dynamic ontology
                     └─> Candidate pairing                      blocking + similarity
                         └─> Adjudication  rules first, LLM only for ambiguous pairs
                             └─> SQLite (facts / evidence / relations)
                                 └─> FastAPI + minimal UI
```

### Decisions to defend in the README
- **SQLite, not Neo4j.** They devalued the graph explicitly. Relational + a `relations` table does
  everything needed, runs from `git clone` with zero setup.
- **Deterministic normalization; LLM only for judgment.** Never ask an LLM to convert crore→million.
  Unit math must be verifiable and testable.
- **Provider-agnostic LLM layer.** Gemini free tier default, Groq/Claude swappable. Graders need no
  paid account.
- **Two-stage extraction** (cheap page filter → expensive claim extraction) handles 100-page PDFs.
- **Precision over recall.** 40 well-grounded facts beat 400 mushy ones.

### LLM provider
| Provider | Model | Cost | Role |
|---|---|---|---|
| Google Gemini (AI Studio) | `gemini-2.5-flash` | **free** | default |
| Groq | `llama-3.3-70b-versatile` | free | fallback |
| Anthropic | `claude-opus-5` | paid | optional, highest quality |

---

## 5. Build Checklist

Agile: every step ends with a runnable test before moving on.

### Phase 0 — Scaffold
- [x] Read assignment, mine all 6 PDFs, verify the four required cases exist
- [x] Choose stack (Python + FastAPI + SQLite) and free LLM provider (Gemini)
- [x] Write this plan
- [x] Repo scaffold, `.gitignore`, `requirements.txt`, `.env.example`
- [x] `git init` + first commit

### Phase 1 — Grounding layer (no LLM yet)
- [x] `pdf_parser.py` — extract text per page with char offsets and bboxes
- [x] `segmenter.py` — split into evidence units (paragraph / table row)
- [x] **TEST:** given a quote, resolve it back to page + bbox in the source PDF
- [x] **TEST:** all 6 starter PDFs parse without error, report unit counts

### Phase 2 — Extraction
- [x] `llm/provider.py` — provider abstraction (Gemini / Groq / Anthropic)
- [x] `schema.py` — typed Fact model with context keys
- [x] `extractor.py` — evidence unit → list of Facts, structured output
- [x] **TEST:** run on Q4 deck (27pp, cheapest); inspect claim quality by hand
- [x] **TEST:** every extracted fact's quote is verbatim-present in its evidence unit

### Phase 3 — Normalization
- [x] `normalize/units.py` — crore/lakh/million/billion, ₹/$/%
- [ ] `normalize/periods.py` — FY24, FY2024/25, Q4 FY24, "as of Dec 31 2021"
- [ ] `normalize/entities.py` — alias resolution
- [x] **TEST:** unit table — 8,142 Cr == 81,420 Mn (23 tests)
- [x] **TEST:** period table — FY24 == FY2023-24, Q4 FY24 nested in FY24

### Phase 4 — Reasoning engine
- [x] `pairing.py` — candidate fact pairs via subject+metric blocking
- [x] `adjudicate.py` — the context-key rule + LLM for ambiguous pairs
- [x] **TEST:** Case 2 (standalone vs consolidated) → RECONCILED, not CONTRADICTS
- [x] **TEST:** Case 1 (Mn vs Cr) → CORROBORATES
- [x] **TEST:** Case 3 (GDP vintage) → RECONCILED
- [x] **TEST:** Case 4 (director) → temporal, both valid

### Phase 5 — Storage + API + UI
- [ ] `db.py` — SQLite schema: documents, evidence, facts, relations
- [ ] `main.py` — FastAPI: POST /documents, GET /facts, GET /relations, GET /evidence/{id}
- [ ] Minimal UI: upload, fact table, evidence panel, relation view with reasoning
- [ ] **TEST:** upload a PDF the system has never seen; verify end-to-end

### Phase 6 — Ship
- [x] Extraction cache (content-addressed; makes incremental ingest work)
- [ ] Incremental ingest (new doc compares only against candidates) — brownie point
- [ ] README: Setup, Video Demo, Approach, Limitations & Next Steps, Additional Notes
- [ ] Record 3-minute demo video (scripted: 20s arch / 30s ingest / 90s four cases / 20s limits)
- [ ] Final commit, push, submit form by 10:00 AM 09-09

---

## 6. Explicitly Cut

Auth, Docker, broad test coverage, pretty graph visualization, multi-user, high extraction recall.
Every one of these trades against the things Superjoin said they would actually grade.
