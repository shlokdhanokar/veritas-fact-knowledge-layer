# Veritas — a Fact Knowledge Layer

Upload PDFs. Get facts that are traceable to the exact page and characters they
came from, and judgements about how those facts relate: **corroborated**,
**contradicted**, or **reconciled by context**.

Built for the Superjoin VIT 2026 Engineering Intern assignment.

---

## Setup and Run Instructions

Requires Python 3.11+ and a free Google Gemini API key
([aistudio.google.com/apikey](https://aistudio.google.com/apikey) — no credit card).

```bash
git clone <this-repo>
cd veritas
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env        # then paste your key into GEMINI_API_KEY

uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000>, drop a PDF on the sidebar, and watch it ingest.
A 100-page filing takes a few minutes; the UI polls and reports progress.

**No paid service is required.** Gemini's free tier runs the whole system. Groq
(also free) and Anthropic are drop-in alternatives — change `LLM_PROVIDER` in
`.env` and nothing else.

### Command line

```bash
python scripts/ingest.py path/to/doc.pdf --limit 40 --out data/facts.json
python scripts/relate.py data/facts.json --verdict CONTRADICTS
python -m pytest -q                                   # 110 tests, no API key needed
```

`--limit` caps how many passages are read per document. Leave it off to read
everything; set it low to stay inside a tight free-tier quota.

### API

| Endpoint | Purpose |
|---|---|
| `POST /api/documents` | Upload a PDF. Returns a job id. |
| `GET /api/jobs/{id}` | Ingest progress. |
| `GET /api/facts` | Facts, filterable by document, text, confidence. |
| `GET /api/relations` | Judgements with both facts and their evidence inlined. |
| `GET /api/stats` | Corpus counts and verdict breakdown. |
| `POST /api/relations/rebuild` | Re-adjudicate after a rules change. |

---

## Video Demo

**<!-- PASTE YOUR LINK HERE -->**

---

## Approach

### The problem with the obvious data model

The natural way to model a fact is `(subject, metric, value)`. That model is
broken in a way that only shows up on real documents.

Delhivery's FY24 annual report states revenue as **₹74,540.82 million** *and*
**₹81,415.38 million**, on the same page, for the same company and the same
year. Both are correct. They differ because one is standalone and the other
consolidated. A `(subject, metric, value)` system has nowhere to put that
distinction, so it must report a contradiction that does not exist.

### Context keys

So a fact here is a value plus **the qualifiers that make it comparable**:

```
subject · metric · value · unit
+ context:   period, basis, entity_scope, vintage, geography, segment, measure
+ evidence:  doc_id, page, char span, bounding boxes, verbatim quote
+ validity:  valid_from, valid_to
```

`context` is an open dictionary, not fixed columns, so a new kind of document
can introduce a qualifier nobody anticipated without a schema migration.

The comparison engine then asks **one question**, and all three required
relationships fall out of the answer:

| Context keys | Values | Verdict |
|---|---|---|
| match | agree | **CORROBORATES** |
| match, all stated on both sides | disagree | **CONTRADICTS** |
| match, but one side leaves a qualifier unstated | disagree | **LIKELY_CONTRADICTS** |
| differ | either | **RECONCILED** — naming the key that explains the gap |

One mechanism, four outcomes, no per-case special-casing.

### Pipeline

```
PDF ─▶ PyMuPDF parse          text + page + char offsets + bounding boxes
    ─▶ page-scale windows     scored for fact density, densest read first
    ─▶ LLM extraction         typed schema, structured output, cached
    ─▶ GROUNDING GATE         quote must verify against source, or the fact dies
    ─▶ normalization          units, magnitudes, fiscal periods  [deterministic]
    ─▶ blocking + adjudication rules first, arithmetic where possible
    ─▶ SQLite ─▶ FastAPI ─▶ UI
```

### Decisions and trade-offs

**Grounding is enforced, not requested.** The model returns a verbatim quote;
we search for it in the source window. Not found means not grounded means the
fact is dropped. Fabricated evidence cannot enter the store, because the check
is mechanical rather than a matter of trusting the model. On the starter set
this rejected between 2% and 41% of proposals depending on the document — the
RBI report's statistical appendices being the hardest.

**The LLM never does arithmetic.** It reports what a document *says*
("8,142", "Cr", "₹"); tested code decides what that *means*. Unit conversion is
the one place where being subtly wrong is invisible and poisons every downstream
comparison, so it is plain code with a test table — including Indian magnitudes,
where a system that only knows million/billion silently mangles every filing.

**SQLite, not a graph database.** The brief says outright that a graph database
is not the solution, and it is right: relations here are pairwise judgements,
which is a table with two foreign keys. The payoff is that the project runs from
a clean clone with nothing to install.

**Precision over recall.** Passages are scored by a generic fact-density
heuristic and the densest are read first. Reading forty passages carefully beats
reading two hundred badly, and the brief asks for useful grounded facts rather
than coverage.

**Bigger windows turned out to be both cheaper and more accurate.** At ~3,000
characters the model saw table rows stripped of their headers and grounded
75–81% of proposals. At page scale it can see the header that says
"Standalone | Consolidated", and grounding rose to 96–100% while using seven
times fewer requests. This was measured, not predicted — the opposite of what I
expected.

**Free-tier quota shaped the architecture.** The measured limit was 20 requests
per day *per model*, so the provider's fallback chain doubles as a quota pool
across eight models, and the content-addressed extraction cache is load-bearing
rather than an optimisation. Because that cache stores raw model output, the
entire corpus can be re-normalised after a code change in under a second with no
API calls at all — which is how several of the fixes below were validated.

**Incremental ingest** falls out of the design: documents are content-addressed
by their bytes, and only pairs touching a new fact are adjudicated. Adding a
document costs that document, not the corpus.

### AI tools used

Claude Code (Claude Opus 5) for implementation throughout. Google Gemini
(`gemini-3.6-flash`, falling back through seven other models) for fact
extraction at runtime. Every design decision, and every fix listed below, came
from running the system on the real documents and reading the output.

### Three bugs the real data found

These are worth reporting because none were visible from the design, and each
changed the system:

1. **89 false contradictions in one earnings deck.** Token-overlap similarity
   treated `net cash from operating activities` and `...investing activities` as
   the same metric — five of six words shared. Requiring that two facts' metric
   *and subject* differ by nothing meaningful took cross-document false
   contradictions from **311 to 1**. Component labels (`(B)` vs `(A+B)`) are
   parsed directly, since a word tokenizer discards them as noise.

2. **A silent recall failure in units.** The model puts magnitude inside the
   unit field (`"Indian Rupees in million"`, `"₹ Cr"`) and abbreviates INR to
   `"I"`, fragmenting one currency across five incomparable spellings. Facts that
   should have corroborated were never even compared. Unit phrases are now
   parsed rather than looked up, and an unusable unit is read off the evidence
   quote instead — grounded inference rather than a guess.

3. **The density heuristic preferred tables to prose.** Headline claims are
   usually sentences ("GDP growth for FY25 is estimated to be 6.4 per cent"), but
   digit-density scoring ranked those pages 58/89 and 83/100, below far less
   meaningful appendix tables. Assertive-claim sentences are now scored
   explicitly.

---

## The four required cases

### 1. A fact corroborated across documents, expressed differently

| | |
|---|---|
| Q4 FY24 earnings deck, p6 | `₹127Cr / 1.6%` |
| Annual report FY24, p4 | `₹1,266Mn` |

Both normalize to **1.27 × 10⁹ INR**, 0.315% apart — within rounding tolerance.
Different documents, different units, different phrasing, same fact. The system
shows the arithmetic that proves it.

### 2. A genuine or likely contradiction

The engine reports **CONTRADICTS** only when every qualifier that could explain
a gap is stated on both sides and agrees. Where one side is silent it reports
**LIKELY_CONTRADICTS**, names the unstated qualifier, and lowers confidence —
because a difference that a missing qualifier might explain is not yet a
confirmed error.

The surviving confirmed contradiction in the Delhivery corpus is two different
amounts paid under protest, extracted from the same Spoton service-tax passage —
which is itself an extraction ambiguity, and is discussed in case 4.

### 3. An apparent contradiction explained by context

| | |
|---|---|
| Annual report FY24, p22 | Revenue **₹74,540.82M**, `basis=standalone` |
| Annual report FY24, p22 | Revenue **₹81,415.38M**, `basis=consolidated` |

An 8% gap that a naive system reports as a contradiction. Veritas returns
**RECONCILED**, names `basis` as the differing key, and explains that a
consolidated figure includes subsidiaries a standalone figure excludes.

The same mechanism handles **valid time**: the prospectus lists Sandeep Kumar
Barasia as an Executive Director, while the FY24 report records him ceasing on
2024-07-01. Not a contradiction — a timeline, where the later state supersedes
the earlier one.

### 4. An extraction or reasoning failure, and how it is handled

**Table column binding.** Financial tables linearise to a flat run of numbers:

```
Line haul expenses  724  616  696  708  2,517  2,684
```

Under a two-level header (`Standalone | Consolidated` × `FY24 | FY23`) the value
is real but *which column it belongs to* is inferred. The model answers
confidently either way, and nothing in the output reveals the doubt. This is the
most dangerous failure in the system because it is silent and plausible.

**How it is handled:** a quote carrying many candidate numbers is weak evidence
for any single one of them, so confidence is cut proportionally and a note
records that the binding was inferred. Roughly a third of extracted facts carry
this flag, and it is visible in the UI on every affected fact.

**What I would do instead, with more time:** read the table's geometry rather
than its text. PyMuPDF already gives bounding boxes for every span, so a value's
column can be determined by x-coordinate against the header row's boxes, turning
an inference into a measurement. The parser records the boxes; only the
table-reconstruction step is missing.

---

## Limitations and Next Steps

**What does not work yet**

- **No OCR.** Scanned PDFs with no text layer are rejected with a clear message
  rather than silently producing nothing.
- **Table structure is not reconstructed**, which is the root of the column
  binding problem above.
- **Coverage is bounded by the reading budget.** With `--limit` set, low-density
  pages are never read, so a fact stated once in unremarkable prose can be
  missed. The scorer improvement helped but did not eliminate this.
- **Synonym handling is a small hand-written list.** `revenue`/`income` unify;
  an unlisted pair of synonyms will be treated as different metrics — which fails
  safe (no contradiction claimed) but costs recall.
- **Entity resolution is token-based.** "Delhivery" and "Delhivery Limited"
  unify; a genuine alias like a renamed subsidiary would not.
- **No LLM adjudication path is wired up.** The `method="llm"` branch exists in
  the schema for pairs rules cannot settle, but every verdict currently comes
  from rules or arithmetic. In practice this was a feature, not a gap — the
  deterministic path is auditable — but genuinely ambiguous metric equivalence
  would benefit from it.

**What I would build next, in order**

1. **Geometry-aware table reading** — the highest-value fix, since it converts
   the system's worst failure mode from a guess into a measurement.
2. **Embedding-based metric canonicalisation** to replace the synonym list, with
   the rule guard kept as a safety net.
3. **A corpus-level hypothesis check for `LIKELY_CONTRADICTS`**: if one side
   omits `basis`, look for a sibling fact in the same document that supplies it.
   The knowledge layer can often resolve its own ambiguity.
4. **Evidence highlighting in a rendered PDF view.** Bounding boxes are already
   stored per fact; only the viewer is missing.

---

## Additional Notes

**On honesty in the output.** The number I would most want a reviewer to look at
is not the fact count but the *drop* count. Every ingest reports how many
proposals were rejected as ungrounded, and how many surviving facts carry an
ambiguity flag. A system that reports 900 facts and 0 problems is not being
truthful about what reading a 100-page filing is actually like.

**On the four cases.** They are also executable tests
(`tests/test_adjudicate.py`), not just screenshots — including the negative
cases that keep the system honest: that an absent qualifier is not treated as a
difference, that a quarter never corroborates its own fiscal year, and that a
genuine contradiction is still reported when nothing explains it.

**Repository layout**

```
app/
  parsing/     PDF → text with char-offset → geometry index; windowing
  extraction/  prompt, LLM extraction, grounding gate, cache
  normalize/   units, magnitudes, currencies         [deterministic, tested]
  reasoning/   pairing, adjudication, the one rule
  web/         UI
  schema.py    the fact model
  store.py     SQLite
  pipeline.py  ingest orchestration, shared by CLI and API
scripts/       ingest, relate, load
tests/         110 tests, none requiring an API key
```

**Credentials** are read from `.env`, which is gitignored. No key is committed.
