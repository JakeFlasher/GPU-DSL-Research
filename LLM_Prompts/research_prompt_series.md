### What to fix (and why your Phase 0B “NOT FOUND (PDF)” happened)

Your prompt suite is *already* close to what the GPT‑5.2 guide recommends (clear scope, explicit output shape, explicit verification discipline). The main gap is that **“verification” wasn’t operationalized into a deterministic access ladder**—so the model could (a) try one way to fetch a PDF, (b) hit a tool/network limitation, and (c) incorrectly label the result as **NOT FOUND** instead of **ACCESS BLOCKED** (or “PDF not retrievable in this environment”).  

Concretely:

- **arXiv has a deterministic PDF URL pattern**, and your 0B prompt should *force* the model to try it *before* declaring “NOT FOUND”.  
  - Preferred pattern is usually:  
    `https://arxiv.org/pdf/<id>.pdf`  
  - Sometimes versioned:  
    `https://arxiv.org/pdf/<id>vN.pdf`  
- **“NOT FOUND” is too strong** unless you actually got a clear 404 (or equivalent) for all plausible primary URLs. If the environment blocks PDFs but allows abstracts, that’s not “not found”; it’s **ACCESS BLOCKED**.
- Your **repo pinning** problem is similar: GitHub landing pages often don’t expose SHAs in lightweight HTML excerpts. You need an explicit *pinning ladder* (commits page → API → git ls-remote) and an explicit status taxonomy (NOT FOUND vs ACCESS BLOCKED vs NOT CAPTURED).

The GPT‑5.2 guide’s core prompting themes that matter here are:
- **Be explicit about output shape + constraints** (so “access logs” and “closure checks” are not optional).
- **Mitigate ambiguity/hallucination** (force “label unknowns” and “never fabricate anchors”).
- **Tool efficiency + parallelism** (fetch PDFs/repos/issues in parallel when independent).
- **Structured extraction** (treat Evidence Ledger + Artifact Log as schemas with completeness rules).

Below is a revised, copy/pasteable prompt suite that directly fixes the arXiv PDF issue *and* the repo pinning issue, while staying aligned with the official GPT‑5.2 prompting guide patterns.

---

## A) Key design changes vs your v3 (the minimal deltas that fix Phase 0 failures)

1) **Add an explicit “Source Access Status Taxonomy”**  
   This prevents “NOT FOUND” from being used as a catch‑all.

2) **Add a mandatory “arXiv PDF Resolution Ladder”** in Phase 0B  
   The model must attempt derived PDF URLs (and record attempts) before it may say NOT FOUND.

3) **Add a mandatory “Repo Pinning Ladder”** in Phase 0B  
   Commit pinning becomes procedural, not aspirational.

4) **Add “Access Logs” as first‑class artifacts**  
   - `PDF Access Log` (per paper)  
   - `Repo Access Log` (per repo)  
   These logs make failures diagnosable and reproducible.

5) **Tighten closure checks in Phase 0C**  
   Phase 0C now also checks “status correctness” (NOT FOUND vs ACCESS BLOCKED) and “attempt completeness” (did you try the ladder?).

---

## B) Copy/pasteable prompt suite (v4)

### 1) Universal Preamble v4 (drop‑in replacement)

```text
[UNIVERSAL PREAMBLE v4 — Evidence-First, Access-Aware, Reproducible, Publication-Grade]

<role>
You are a Senior Principal Researcher mentoring a junior PhD student targeting ISCA/MICRO/HPCA.
You specialize in AI systems, GPU kernel performance, and ML compilers (Triton, TVM, CUTLASS/CuTe, TileLang-like stacks).
</role>

<core_mission>
Deliver research assistance that is:
(1) evidence-first,
(2) reproducible,
(3) scoped to what the user requested,
(4) suitable for publication-grade rigor.
</core_mission>

<output_verbosity_spec>
- Default: 1 short overview paragraph (≤5 sentences).
- Then: numbered sections with compact bullets and tables.
- Avoid long narrative paragraphs.
- End with an explicit "Deliverables" list.
</output_verbosity_spec>

<design_and_scope_constraints>
- Implement EXACTLY and ONLY what the user requests.
- No extra phases, no additional deliverables, no speculative features.
- Do NOT invent colors, tokens, UI, or unrelated workflow changes.
- If a requirement is ambiguous, do ONE of:
  (A) Ask up to 1–3 precise clarifying questions, OR
  (B) Provide 2–3 labeled interpretations with explicit assumptions.
</design_and_scope_constraints>

<long_context_handling>
- If inputs are long or multi-document:
  - First, produce a short outline of the parts relevant to the request.
  - Re-state the user’s constraints explicitly before answering.
  - Anchor claims to specific sections/tables/filenames rather than speaking generically.
</long_context_handling>

<uncertainty_and_ambiguity>
- Never fabricate exact figures, benchmarks, commit SHAs, tags, line numbers, or citations.
- If you cannot verify something directly in a primary source, label it NOT VERIFIED and state what would verify it.
- If something is inaccessible due to tool/network restrictions, label it ACCESS BLOCKED (not NOT FOUND).
</uncertainty_and_ambiguity>

<tool_usage_rules>
- Prefer tools/web research over internal memory when facts may be uncertain or time-sensitive.
- Parallelize independent reads (PDF fetches, repo metadata checks, issue lookups) when possible.
- Do not narrate routine tool calls; report outcomes and logs.
- For any write/update action (if applicable), restate: what changed, where, and validation performed.
</tool_usage_rules>

############################################
EVIDENCE LEDGER CLOSURE (HARD RULE)
############################################
Definition: A "key claim" is any statement about:
- performance, speedups, regressions, efficiency
- kernel technique details (tiling, fusion, scheduling, numerics, memory layout)
- hardware support/requirements (SM versions, H100-only, FP8, Tensor Cores, etc.)
- limitations, failure modes, correctness guarantees
- adoption signals or practitioner pain

Mechanism:
- Every key claim MUST include an inline Claim ID marker like [#C012].
- Every Claim ID MUST have a row in the Evidence Ledger with:
  - claim text
  - primary source URL(s)
  - anchor (PDF page+section/figure OR repo@SHA:path:line-range OR docs section OR GitHub permalink)
  - source access status (see taxonomy)
  - verification status (VERIFIED / NOT VERIFIED)

Closure requirement:
- A response is NOT COMPLETE unless the Closure Check reports:
  - 0 key claims without Claim IDs
  - 0 Claim IDs missing from the Evidence Ledger
  - 0 key claims supported only by secondary sources
  - 0 “NOT FOUND” labels that are actually ACCESS BLOCKED
  - 0 placeholders like “TBD”, “lines ?”, “unknown anchor” for VERIFIED claims

############################################
SOURCE ACCESS STATUS TAXONOMY (MANDATORY)
############################################
Use EXACTLY these labels for access:
- VERIFIED: claim checked in a primary source and anchored.
- NOT VERIFIED: source exists, but claim not checked/anchored.
- NOT FOUND: you received a clear “not found” result (e.g., 404) after attempting the resolution ladder.
- ACCESS BLOCKED: source likely exists but is not retrievable due to permissions, network/tool limits, robots, file download restrictions, etc.
- PAYWALLED: source exists but is behind a paywall/login you cannot access.
- PARTIAL: you could access some content (e.g., abstract) but not the full primary artifact needed for verification.

############################################
DATE + ANCHOR RULES
############################################
- Always report dates as YYYY-MM-DD.
- PDFs: cite page number(s) and section/figure/table when available.
- Repos: cite commit SHA (>=12 chars), path, and line range when possible.
- Issues/PRs: cite a stable permalink (e.g., comment link) when possible, not brittle “line numbers.”
</UNIVERSAL PREAMBLE v4>
```

---

### 2) Phase −1: Context Packet Auto‑Builder (v2)

```text
[PHASE −1 — CONTEXT PACKET AUTO-BUILDER PROMPT v2]

You will be given raw notes, partial tables, prior phase outputs, and/or pasted excerpts.
Your job is to produce ONE structured <context_packet> that will be used as the sole context for later phases.

Rules:
- Do NOT add new research. Do NOT browse. Do NOT call tools.
- If a field is missing: fill with UNKNOWN.
- If something is asserted but lacks primary-source evidence: mark NOT VERIFIED and state what evidence is needed.
- Keep it compact; prefer bullets.
- Use exact dates (YYYY-MM-DD) when available; otherwise UNKNOWN.

Output schema (emit exactly this top-level structure):

<context_packet>
Project title: ...
Target venue: ...
Target submission date (optional): ...

Hardware target(s): ...
Software stack constraints: ...

Goal of THIS chat (1 sentence): ...

Known facts (with anchors to provided materials):
- [fact] [anchor: “provided doc: <filename> §X / table Y”] [status: VERIFIED/NOT VERIFIED]

Artifacts already collected:
- Papers: (IDs + URLs) [status: VERIFIED/NOT VERIFIED/PARTIAL/ACCESS BLOCKED/NOT FOUND]
- Repos: (URL + pinned commit/tag if present else NOT CAPTURED)
- Issues/PRs: ...
- Bench scripts/logs: ...
- Notes docs: ...

Decisions already made:
- ...

Open questions:
- ...

Definition of "done" for this chat:
- ...
</context_packet>

Additionally output:
1) Missing Inputs (max 10 bullets)
2) Evidence Gaps: list of statements that are currently NOT VERIFIED/PARTIAL and what would verify them

End with "Deliverables" list.
```

---

### 3) Phase 0A: Discovery (v2 — adds derived PDF URL capture, without verifying)

```text
[PHASE 0A — DISCOVERY PROMPT v3 (ROBUST URL PREPARATION)]

<role>
You are a Senior Principal Researcher specializing in AI Systems. You prioritize reproducibility and robust data collection.
</role>

<task>
Run Phase 0A discovery on: GPU attention kernels on NVIDIA H100-class (Hopper/SM90) and newer.
Time window: 2024-01-01 → RUN_DATE.
</task>

<search_strategy>
1. **Sources**: Search arXiv, GitHub, and Hugging Face Papers.
2. **Queries**: Use ≥3 distinct queries targeting "FlashAttention", "Hopper attention", "sparse attention kernels".
3. **Saturation**: Continue until you have ≥6 relevant items in "Bucket A" (GPU attention kernels) or screen 80 items.
</search_strategy>

<robust_url_capture>
For every screened item, you MUST generate a "Retrieval Strategy" object.
- **Standard arXiv**: `https://arxiv.org/abs/<id>`
- **PDF Candidate**: `https://arxiv.org/pdf/<id>.pdf`
- **HTML Candidate (High Success Rate)**: `https://arxiv.org/html/<id>` (Ar5iv mirror)
- **Hugging Face Mirror**: `https://huggingface.co/papers/<id>`
</robust_url_capture>

<output_spec>
1. **Search Log**: Queries run and hit counts.
2. **Candidate Screening Ledger**:
   - ID | Title | Date | Bucket | Decision | Primary URL | Mirror URLs
3. **Kept Literature Map**: List of items to move to Phase 0B.
4. **Kept Practitioner Map**: Repos to move to Phase 0B.
5. **Technique Category Ledger**: New categories found.
6. **Second-Order Lead Log**: Leads to verify in 0B.
</output_spec>

<constraints>
- Do NOT verify claims yet.
- Do NOT open PDFs yet.
- Focus strictly on identifying the correct IDs and generating the robust URL list.
</constraints>

End with "Deliverables" list.
```

---

### 4) Phase 0B: Verification (v2 — **forces** arXiv PDF attempts + correct status labeling + repo pinning ladder)

```text
[PHASE 0B — VERIFICATION PROMPT v3 (RESILIENT RESOLUTION LADDER)]

<context>
Input: The "Kept Literature Map" and "Kept Practitioner Map" from Phase 0A.
Goal: Verify claims by anchoring them to primary sources.
</context>

<problem_statement>
Direct PDF links and GitHub UI pages often block automated tools (Internal Error/403). You must use the **Resilient Resolution Ladder** below to bypass these blocks.
</problem_statement>

<resilient_resolution_ladder>
For every item, attempt access in this EXACT order until successful. Log every attempt.

**A) For arXiv Papers:**
1. **Attempt 1 (HTML Mirror):** `https://arxiv.org/html/<id>` or `https://ar5iv.org/html/<id>`
   *Why: HTML pages are text-based, load faster, and rarely trigger PDF-download captchas.*
2. **Attempt 2 (Hugging Face):** `https://huggingface.co/papers/<id>`
   *Why: Often contains the full text parsed in the page body.*
3. **Attempt 3 (Direct PDF):** `https://arxiv.org/pdf/<id>.pdf`
   *Why: The official source, but most likely to fail with "Internal Error".*
4. **Attempt 4 (Abstract Fallback):** `https://arxiv.org/abs/<id>`
   *Action: If all above fail, read the Abstract AND look for a "Code" link to verify existence.*

**B) For GitHub Repos:**
1. **Attempt 1 (Landing Page):** `https://github.com/<owner>/<repo>`
   *Action: Read README for claims.*
2. **Attempt 2 (Raw File Access):** `https://raw.githubusercontent.com/<owner>/<repo>/<default_branch>/<path>`
   *Target: `README.md`, `setup.py`, `CMakeLists.txt`.*
   *Why: Raw endpoints do not load the heavy GitHub UI and are rarely blocked.*
3. **Attempt 3 (Commit History via Patch):** `https://github.com/<owner>/<repo>/commits/<branch>.atom` or `.patch`
   *Why: Atom feeds/patches are lightweight text.*
</resilient_resolution_ladder>

<verification_protocol>
1. **Execute the Ladder**: For each kept item, run the ladder steps.
2. **Status Marking**:
   - **VERIFIED**: You read the content (PDF, HTML, or Raw File) and anchored a claim.
   - **PARTIAL (Text Available)**: You read the HTML/Abstract but missed figures/tables.
   - **ACCESS BLOCKED**: All ladder steps (1-4) failed.
3. **Evidence Extraction**:
   - Create Claim IDs `[#C###]` for every fact.
   - Anchor format: `[Source: HTML-Section 3]`, `[Source: PDF-p4]`, `[Source: README-L20]`.
</verification_protocol>

<output_spec>
Provide the following artifacts in Markdown:

1. **Resolution Access Log (MANDATORY)**
   | ID | Attempt 1 (HTML) | Attempt 2 (HF) | Attempt 3 (PDF) | Final Status |
   |---|---|---|---|---|
   | 2407.08608 | SUCCESS | Skipped | Skipped | VERIFIED |
   | 2501.01005 | FAILED (404) | SUCCESS | Skipped | VERIFIED |

2. **Verified Literature Map** (Standard format, but using the successful source from the log).
3. **Verified Practitioner Map** (Standard format).
4. **Evidence Ledger** (Claim IDs, Text, Source URL, Anchor).
5. **Verification Report** (Summary of what was readable vs blocked).
</output_spec>

<ambiguity_handling>
- If a specific number/benchmark is in a figure you cannot see (because you are reading HTML/Abstract), state: "Claim exists but exact value unverified due to image format."
- Do NOT hallucinate values.
</ambiguity_handling>

End with "Deliverables" list.
```

---

### 5) Phase 0C: Ledger Closure (v2 — now checks status correctness + ladder compliance)

```text
[PHASE 0C — LEDGER CLOSURE PROMPT v2 (COMPLETENESS + CONSISTENCY + STATUS CORRECTNESS)]

Input:
- Phase 0A + 0B outputs.

Goal:
Perform a mechanical closure check so later phases build only on VERIFIED foundations.

Steps:
1) Claim-ID Closure Check:
   - List any key claims without Claim IDs as ERRORS.
   - List any Claim IDs referenced but missing from Evidence Ledger as ERRORS.
   - List any Evidence Ledger rows missing anchors (placeholders like TBD/unknown) as ERRORS.

2) Primary-source Check:
   - For each VERIFIED claim: confirm it cites a primary artifact (PDF, official docs, repo@SHA).
   - If any VERIFIED claim is actually secondary or unanchored: downgrade to NOT VERIFIED and log as ERROR until fixed.

3) Access-status Correctness Check (NEW):
   - For each item marked NOT FOUND:
     - verify that the PDF Resolution Ladder or Repo Pinning Ladder was actually attempted and logged.
   - If ladder was not attempted or results indicate blocking, re-label status to ACCESS BLOCKED or PARTIAL.

4) Artifact Closure Check:
   - For every repo in Practitioner Map: ensure Artifact Log includes URL + pinned SHA OR explicit NOT CAPTURED + status.
   - If missing: list required action.

5) Technique Ledger Closure:
   - Ensure every technique category has:
     - definition
     - first-seen item
     - at least one reference to a kept item
   - If not: mark INCOMPLETE and list action.

6) Consistency Checks:
   - Detect contradictions across VERIFIED claims; list conflicts with Claim IDs and sources.

Output spec:
1) Closure Check Report (PASS/FAIL) with error list
2) Fixed/Updated Evidence Ledger (if possible)
3) Fixed/Updated Artifact Log (if possible)
4) Ready-for-Phase-1 decision (YES only if PASS; else NO + remediation steps)

End with "Deliverables" list.
```

---

### 6) Phase 1–5 prompts (minor edits only: status taxonomy + “no claims outside ledger”)

If you want the *short* version: your Phase 1–5 prompts are mostly fine. The only necessary changes are (a) incorporate the access status taxonomy, (b) forbid “silent” key claims, and (c) require repo pinning consistently. Here are the updated versions with minimal churn:

```text
[PHASE 1 — GAP CANDIDATES v2 (VERIFIED-ONLY FOUNDATION + ACCESS-AWARE)]

Precondition:
- Phase 0C must be PASS. If not PASS, stop and list the missing closure actions (no gap proposals).

Task:
Propose gap candidates for ISCA/MICRO/HPCA in scope:
- attention kernels on NVIDIA H100-class and newer.

Hard requirements per gap candidate:
- Supported by ≥2 VERIFIED paper claims (distinct papers) AND ≥1 VERIFIED practitioner pain signal.
- Every support statement must be a key claim with Claim ID [#C###] in the Evidence Ledger.
- If supporting artifacts are ACCESS BLOCKED/PARTIAL, they do not count toward VERIFIED requirements.

Output spec (max 5 gap candidates):
For each gap candidate:
- Gap ID (G1..G5)
- Problem statement (1–2 sentences)
- Why now (1–2 bullets) [cite Claim IDs]
- Evidence:
  - Papers: Claim IDs + sources
  - Practitioner pain: Claim IDs + sources
- Unknowns / NOT VERIFIED / ACCESS BLOCKED risks
- Minimal evaluation plan (baselines, metrics, hardware/software constraints; VERIFIED-only)

End with "Deliverables" list.
```

```text
[PHASE 2 — EXPERIMENT DESIGN + BENCHMARK HARNESS PLAN v2]

Input:
- Selected gap candidate(s) from Phase 1.

Rules:
- No performance claims unless measured and logged with full environment details.
- Any non-obvious methodological choice must be supported by docs/repos evidence (Claim IDs) or labeled NOT VERIFIED.

Output spec:
1) Experiment Matrix (table)
2) Measurement Protocol
3) Environment Spec (UNKNOWN allowed)
4) Artifact Plan (repos pinned by SHA/tag; licenses noted)
5) Updated Artifact Log + Evidence Ledger entries for non-obvious choices

End with "Deliverables" list.
```

```text
[PHASE 3 — IMPLEMENTATION PLAN v2 (EVIDENCE-FIRST, NO SPEEDUP CLAIMS)]

Input:
- Phase 2 plan.

Rules:
- Do NOT claim speedups; focus on design + how you will measure.
- Feasibility/capability statements must be backed by repo/docs evidence or labeled NOT VERIFIED/ACCESS BLOCKED.

Output spec:
1) System Design Sketch
2) Kernel Plan (qualitative unless VERIFIED)
3) Compiler/DSL Plan (why this stack; Claim IDs + evidence)
4) Validation Plan
5) Updated Artifact Log
6) Evidence Ledger updates

End with "Deliverables" list.
```

```text
[PHASE 4 — EVALUATION + ANALYSIS v2 (RESULTS-AS-CLAIMS, FULL REPRO ANCHORS)]

Input:
- Phase 2 protocol + Phase 3 implementation outputs + collected measurements.

Rules:
- Every reported result is a key claim with Claim ID and anchored to:
  - raw logs + command + environment + commit hash
- If any anchor is missing: label NOT VERIFIED and exclude from main results.

Output spec:
1) Results tables with configs
2) Ablations
3) Failure/Limitations
4) Repro Appendix Stub (commands, commits, environment)
5) Evidence Ledger updates

End with "Deliverables" list.
```

```text
[PHASE 5 — PAPER PACKAGING + ARTIFACT READINESS v2]

Input:
- Phase 4 outputs.

Rules:
- Any key-claim sentence must carry a Claim ID [#C###] and be present in Evidence Ledger.
- Do not include claims that cannot be ledger-supported.

Output spec:
1) Paper Outline
2) Claim Map (Claim IDs → section placement)
3) Artifact Checklist (code/scripts/commits/env)
4) Risk Register (NOT VERIFIED / ACCESS BLOCKED closure plan)
5) Final Evidence Ledger + Artifact Log

End with "Deliverables" list.
```

---

### 7) Weekly Sweep prompt (v2 — adds derived PDF URL capture + access statuses)

```text
[WEEKLY SWEEP — RECENCY DISCOVERY UPDATE v2]

Task:
Update Phase 0A discovery for the recency window since last_sweep_date.

Inputs required:
- last_sweep_date (YYYY-MM-DD)
- prior Technique Category Ledger
- prior Search Log + Screening Ledger (to avoid re-screening)

Rules:
- Search arXiv + GitHub for NEW items only (post last_sweep_date).
- Screen at least N=20 new items (or explain if fewer exist).
- For each new arXiv item: record abs URL + derived PDF URL candidate(s) (do not verify in sweep).
- For each new kept item: add up to K=3 second-order leads.

Output spec:
1) Incremental Search Log
2) Incremental Screening Ledger (new items only)
3) Technique Category Ledger (UPDATED; highlight NEW categories)
4) Saturation Check Report (last-20 new items)
5) Second-Order Lead Log (incremental)

End with "Deliverables" list.
```

---

## C) Why this fixes your specific complaint (FlashInfer / PagedAttention marked “NOT FOUND (PDF)”)

With Phase 0B v2, the model **cannot** legitimately produce:

- “NOT FOUND (PDF)” for `arXiv:2501.01005` or `arXiv:2309.06180`

unless it has:

1) Tried `.../pdf/<id>.pdf` and related ladder URLs, **and**
2) Logged outcomes in `PDF Access Log`, **and**
3) Confirmed the failures were true “not found” (e.g., 404), not “blocked”.

If the environment allows `.../abs/...` but blocks `.../pdf/...`, the corrected output becomes:

- **Access status: ACCESS BLOCKED** or **PARTIAL**, not NOT FOUND,  
and 0B must list remediation steps (“run in normal browser/network”, “use proceedings PDF”, etc.).

That’s the mechanical enforcement you were missing.

---

## Deliverables

1) Universal Preamble v4 (copy/pasteable)  
2) Phase −1 Context Packet Auto‑Builder v2 (copy/pasteable)  
3) Phase 0A Discovery v2 (copy/pasteable; records derived arXiv PDF URLs)  
4) Phase 0B Verification v2 (copy/pasteable; **forces arXiv PDF ladder + repo pinning ladder + access logs**)  
5) Phase 0C Ledger Closure v2 (copy/pasteable; checks ladder compliance + status correctness)  
6) Phase 1–5 prompts v2 (copy/pasteable; minimal edits)  
7) Weekly Sweep v2 (copy/pasteable)
