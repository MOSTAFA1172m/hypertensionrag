# Business Model Brief & Market Value Proposition

**Hypertension Guideline RAG** — AI Clinical Decision Support (CDS)
Prepared: August 2026 · AI Clinical Decision Support Hackathon

---

## 1. Executive Summary

Hypertension affects **1.3 billion people** worldwide and is the leading modifiable risk
factor for cardiovascular death. National and international guidelines — WHO, ESC, USPSTF
— are long, dense, and change frequently. Clinicians do not read them at the point of care,
and there is currently **no free, grounded, guideline-specialist assistant** that answers
clinical questions with **zero hallucination**.

We built exactly that: a retrieval-augmented generation (RAG) system over the WHO 2021,
ESC 2024, USPSTF, and CDC hypertension guidelines, evaluated on a 200-question clinical
benchmark with **faithfulness 5.0/5 (no hallucinations)**, **correctness 4.84/5**, and
**75.5% top-1 retrieval accuracy** on the deduplicated corpus. It is deployed today on a
free Hugging Face Space and answers clinical questions with inline, verifiable citations.

**Value proposition in one sentence:** *Evidence-grounded, citation-backed hypertension
guidance at the point of care, with zero hallucination and near-zero marginal cost.*

---

## 2. Problem & Market

### 2.1 The clinical problem
- Hypertension is treated in **every primary-care clinic in the world**, yet treatment
  decisions vary widely and deviate from guidelines (BP thresholds, first-line drug class,
  comorbidity handling).
- Guidelines are **hundreds of pages, PDF-based, and updated every 1–5 years**. A busy
  clinician cannot recall or search them reliably mid-consultation.
- Generic LLMs **hallucinate** medical guidance — unacceptable in a clinical setting.
  Trust requires that every answer cite the exact guideline section.

### 2.2 Market size
| Segment | Estimate |
|---|---|
| Global hypertension management / digital health (CDS) | Multi-billion $, growing >10%/yr |
| Primary-care clinicians needing hypertension support | Tens of millions worldwide |
| Low- and middle-income countries (WHO 2021 guideline is targeted at these) | Largest underserved segment |
| Medical education / guideline-training market | Large, recurring |

### 2.3 Market gap
- General LLMs: fluent but **not grounded** → rejected for clinical use.
- Static PDFs / guideline apps: grounded but **not queryable**, no clinical UX.
- Enterprise CDS (Epic, etc.): expensive, locked-in, not accessible to smaller clinics or LMICs.
- **Our niche: free-to-try, citation-grounded, specialist assistant with a deployable API.**

---

## 3. Value Proposition

| Stakeholder | Value we deliver |
|---|---|
| **Clinicians** | Instant, cited answers at point of care; correct refusals instead of guesses; saved search time |
| **Health systems / payers** | Adherence to evidence-based treatment → better outcomes, fewer events, lower cost |
| **Medical educators / trainees** | Self-study QA over the actual guideline text |
| **Governments / NGOs (LMIC)** | Free-tier access to WHO-guideline-grounded support where specialist capacity is scarce |
| **Device / EMR vendors** | Embeddable, per-call API to add guideline grounding to their products |

**Differentiation:** *Faithfulness is a product feature, not an accident.* The system is
trained-tuned to refuse when the evidence is absent (correctness failures are clean
refusals, never fabrication) — the only acceptable behavior for clinical decision support.

---

## 4. Product Overview

- **What:** RAG over 4 authoritative hypertension guidelines (WHO, ESC, USPSTF, CDC).
- **Retriever:** BM25 on a deduplicated 435-chunk corpus — fastest and best-ranked of all
  12 variants tested (beats dense, hybrid, and medical cross-encoder rerankers).
- **Generator:** Google Gemini Flash Lite, temperature 0.1, grounded prompt.
- **Trust features:** inline numbered citations mapped to section + page; refusal on
  missing evidence; per-question source cards.
- **Validation:** 200-question benchmark (hit@1 75.5%, MRR 0.826, faithfulness 5.0/5,
  correctness 4.84/5, citation validity 0.99).
- **Deployment:** live Gradio UI (Hugging Face Space) + REST API, Python, ~2 ms retrieval.

---

## 5. Business Model

**Primary: B2B SaaS + API licensing** (recurring, scalable).
**Secondary: Freemium tier** (free to individual clinicians — drives adoption and trust).

| Revenue stream | How it works | Stage |
|---|---|---|
| **API usage (pay-as-you-go)** | Per-call pricing to EMRs, telehealth, pharma content, guideline apps | Core |
| **B2B license (annual)** | Health systems / provider groups: seat-based or facility-based | Core |
| **Education bundles** | Med schools / residency programs: guideline QA & assessment tool | Early |
| **NGO / government contract** | WHO-aligned LMIC deployments (paid, discounted) | Pipeline |
| **White-label deployment** | Vendor ships our engine inside their product | Growth |

### 5.1 Pricing sketch
- **Free tier:** ~5 questions/day per user — acquisition channel (mirrors current HF ZeroGPU deployment).
- **API:** ~$0.02–0.05 per grounded question (covers Gemini + serving + margin).
- **B2B SaaS:** $X per clinician/year or per active user/month, tiered by facility size.
- **White-label:** per-deployment license + support retainer.

### 5.2 Unit economics (illustrative)
| Item | Value |
|---|---|
| Retrieval cost per query | ~$0.00002 (local BM25) |
| Generation cost per query (Flash Lite, ~1.5k tok out) | ~$0.001–0.003 |
| Total marginal cost per question | < $0.005 |
| Per-question price (API) | $0.02–0.05 |
| **Gross margin** | **> 80%** |

---

## 6. Cost Structure
- **Variable (dominant):** LLM inference (Gemini Flash Lite — low).
- **Fixed, minimal:** Hugging Face hosting (free tier today), later GPU/CPU serving.
- **Key asset:** the evaluated, deduplicated guideline corpus + benchmark — built once, reused everywhere (multiple guideline packs = marginal data cost).
- **Enabler of low cost:** local BM25 retrieval means the expensive model is called only for generation, not retrieval.

---

## 7. Go-to-Market

1. **Hackathon/demo** → prove faithfulness + live deployment (done).
2. **Free public Space** → collect real clinical questions, usage patterns, trust signals.
3. **Pilot partnerships** → 2–3 primary-care clinics / telehealth providers.
4. **Content/SEO moat** → guideline-package library (multi-language, multi-country) as a
   content business on top of the engine.
5. **Compliance positioning** → cite faithfulness metrics; pursue medical-device/health-data
   compliance as revenue unlocks (clinical deployment).

---

## 8. Competitive Landscape

| Competitor | Weakness vs. us |
|---|---|
| Generic LLMs (ChatGPT/Gemini) | Hallucination; no citations; not guideline-locked |
| PDF guideline apps / UpToDate-style | Not queryable at point of care; expensive licenses |
| Enterprise EMR CDS | Heavy integration, cost, lock-in; not accessible to small clinics/LMICs |
| Other RAG tools | General-purpose, unvalidated on clinical faithfulness; no clinical benchmark |

**Our moats:** (1) evaluation-driven faithfulness guarantees, (2) the deduplicated
curated corpus, (3) benchmark + test-set as a reusable asset, (4) cheap unit economics.

---

## 9. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| LLM/API cost at scale | Flash Lite tier; cache; batch; local BM25 keeps retrieval free |
| Guideline updates | Corpus pipeline + regeneration; versioned guideline packs |
| Regulatory (medical device) | Position as decision-support reference, not autonomous diagnosis; clinician-in-the-loop |
| ZeroGPU quota / free-tier limits | B2B path moves to paid serving; freemium caps daily free usage |
| Trust failures | Faithfulness-by-design (refuse, don't fabricate); audit trail of citations |

---

## 10. Metrics That Back the Pitch

From `evaluation/` (all reproducible via the benchmark scripts):
- Retrieval: **hit@1 75.5%**, hit@3 89.0%, hit@5 94.5%, **MRR 0.826**, **2 ms/query**
  (BM25, dedup corpus, 200 questions).
- Generation: **Faithfulness 5.0/5**, **Correctness 4.84/5**, **Citation validity 0.99**.
- Dedup was the single biggest lever: +17.5 pts hit@1.

---

## 11. Roadmap (next 12 months)

| Quarter | Milestone |
|---|---|
| Q1 | More guideline packs (multiple languages, more countries) + open benchmark |
| Q2 | API + usage billing; first clinic pilot |
| Q3 | Telehealth / EMR integrations; white-label offer |
| Q4 | LMIC/NGO deployment program; compliance groundwork |

---

*Source of all performance numbers: `evaluation/EVALUATION_REPORT.pdf` and
`evaluation/EXPERIMENTS.md`.*