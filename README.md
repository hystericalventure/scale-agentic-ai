# SCALE: A Consent-Aware Agentic AI Framework for Scaffolded, Disability-Inclusive Education

### Consent-First, Zone-of-Proximal-Development-Based Agentic AI for Disability-Inclusive Learning

SCALE is a multi-agent AI framework for disability-inclusive education, built around three
principles largely missing from prior assistive AI designs: **consent-first adaptation**,
**Zone of Proximal Development (ZPD)-based scaffolding depth**, and **proactive (not reactive)
overload prediction**. This repository contains the full working implementation, experiment
code, results, and an interactive dashboard for the paper submitted to ICADCML 2027.

> **Honesty note:** every result below is produced by the code in this repository, run on
> synthetic data grounded in real statistics from the OULAD dataset. These are
> simulation-based algorithmic findings, not evidence of real-world educational
> effectiveness. See [Limitations](#limitations).

---

## 🚀 Live Dashboard

**🔗 Try it here:** [https://hystericalventure.github.io/scale-agentic-ai/](https://hystericalventure.github.io/scale-agentic-ai/)

Or run it locally — no install needed:
```bash
git clone <this-repo-url>
open scale-agentic-ai/index.html   # or just double-click the file
```

The dashboard lets you:
- Step through a real flagged student's session-by-session agent decisions, live
- Browse all result figures and the 4 evaluation parameters
- See the full system architecture and Consent-Aware PRAC loop diagrams

---

## 📂 Sample Data

- `data/student_profiles.csv` — 400 synthetic student profiles (6 disability sub-types)
- `data/oulad_summary_stats.json` — real statistics computed from OULAD (32,593 real students)

OULAD itself is not redistributed here (download from
[analyse.kmi.open.ac.uk/open_dataset](https://analyse.kmi.open.ac.uk/open_dataset) to
reproduce the grounding step).

---

## 🧠 Framework Overview

SCALE integrates **reinforcement learning, classical machine learning, and rule-based
consent logic** — deliberately *not* an LLM — to make personalized, explainable, and
consent-respecting decisions for disabled and neurodivergent learners.

### Key Components

1. **ZPD Scaffolding Agent**
   - Tabular Q-learning over a 45-state space (mastery × fatigue × recent error)
   - Chooses scaffolding depth: independent work, a partial hint, or a full worked example

2. **Consent & Trust Agent**
   - Gates every proposed change through an explicit accept/reject step
   - Maintains a per-student trust profile that shapes future proposals

3. **Overload Prediction Agent**
   - RandomForest classifier forecasting overload **one session ahead**
   - Triggers a preemptive break before performance degrades, not after

4. **Teacher Co-Pilot Agent**
   - Generates a data-grounded rationale for every decision
   - Flags students needing attention, verified against real outcomes

5. **Consent Ledger (coordination layer)**
   - Reconciles all four agents' outputs
   - Priority: safety/IEP compliance > student consent > pedagogy > engagement nudges

> **No LLM is used or required.** All four agents operate on structured numeric data. The
> architecture includes a planned "LLM Decision Layer" for free-text/voice input in a future
> deployment — designed, not implemented. See [Architecture vs. implementation](#-architecture-vs-implementation).

---

## ✨ Features

- Fully working 4-agent implementation — not pseudocode, real reproducible code
- Closed-loop simulation environment with verified causal dynamics
- Real-data grounding via OULAD (32,593 real students)
- Interactive local dashboard, no server required
- Full statistical rigor: held-out evaluation, multi-seed averaging, significance testing

---

## 🗂 Project Structure

```
scale-agentic-ai/
├── index.html            # Interactive live dashboard — open this first
├── demo.py                   # Live narrated CLI demonstration
├── show_calculation.py       # Transparent walkthrough of how one headline number is computed
├── src/                      # All agent + environment + evaluation code
│   ├── environment.py
│   ├── generate_profiles.py
│   ├── zpd_agent.py
│   ├── overload_agent.py
│   ├── consent_agent.py
│   ├── teacher_copilot_agent.py
│   ├── full_system.py
│   ├── eval_fairness.py          # Evaluation parameter: fairness across sub-types
│   ├── eval_significance.py      # Evaluation parameter: statistical significance
│   ├── eval_learning_efficiency.py  # Evaluation parameter: sessions-to-mastery
│   └── eval_lead_time.py         # Evaluation parameter: overload prediction lead time
├── data/
├── results/                  # All real experiment output (CSVs, figures)
├── ARCHITECTURE.md
└── RESULTS.md
```

---

## ⚙️ Installation

```bash
git clone <this-repo-url>
cd scale-agentic-ai
pip install -r requirements.txt
```

Run the live demo:
```bash
python3 demo.py
```

Reproduce any individual experiment:
```bash
python3 src/zpd_agent.py              # ZPD agent vs. baselines
python3 src/overload_agent.py         # Overload prediction evaluation
python3 src/consent_agent.py          # Consent-gated vs. push-only comparison
python3 src/full_system.py            # Full 4-agent integration (headline results)
python3 src/eval_fairness.py          # Fairness across disability sub-types
python3 src/eval_significance.py      # Statistical significance testing
python3 src/eval_learning_efficiency.py  # Learning efficiency
python3 src/eval_lead_time.py         # Overload prediction lead time
python3 show_calculation.py           # Transparent proof of how a result is calculated
```

---

## 🌐 Deploying the dashboard live (GitHub Pages)

1. On GitHub, go to your repo → **Settings → Pages**
2. Under "Build and deployment", set **Source: Deploy from a branch**
3. Branch: **main**, folder: **/ (root)** → Save
4. Wait 1-2 minutes, then your dashboard is live at:
   `https://YOUR_USERNAME.github.io/scale-agentic-ai/index.html`
5. Paste that link into the "Live Dashboard" section at the top of this README

---

## 📊 Headline Results

All agents trained on 320 students, evaluated on 80 fully held-out students never seen
during training. Reported as mean ± standard deviation across multiple random seeds.

| Condition | Mean reward | Mean overload events | Mean final mastery |
|---|---|---|---|
| **SCALE (full system)** | 1.22 ± 0.27 | 15.28 ± 0.25 | 0.989 ± 0.003 |
| Rule-based baseline | -3.65 ± 0.06 | 22.94 ± 0.35 | 0.816 ± 0.007 |
| Push-only (no consent) | 5.02 ± 0.06 | 14.12 ± 0.40 | 1.000 ± 0.000 |

Full breakdown, per-agent results, and figures: [RESULTS.md](RESULTS.md)

---

## 📐 Evaluation Parameters

Beyond the headline reward/overload/mastery metrics, four additional evaluation
parameters were added for rigor:

| # | Parameter | Result |
|---|---|---|
| 1 | **Statistical significance** (paired t-test vs. rule-based baseline) | All 3 headline metrics significant at p < 0.001 (240 paired observations) |
| 2 | **Fairness across disability sub-types** | Mastery spread only 0.026 across 6 sub-types (very equitable); overload burden varies more, higher for ASD/Physical-Sensory sub-types |
| 3 | **Learning efficiency** (sessions to reach mastery ≥ 0.6) | SCALE: 15.8 sessions vs. rule-based: 25.4 sessions (~38% faster) |
| 4 | **Overload prediction lead time** | Mean 0.61 sessions of sustained prior warning; 27.8% of events get ≥1 session advance warning — reported honestly as a modest result |

Full methodology and figures for each: see the **Evaluation Parameters** tab in
`index.html`, or `src/eval_*.py`.

---

## 🏗 Architecture vs. Implementation

The architecture includes a planned "LLM Decision Layer" for natural-language intent
parsing (Figure 1). **This layer is designed but not implemented.** The four agents that
are implemented communicate via direct function calls on structured data and do not
require an LLM. Adding real chat/Q&A capability would require wiring an LLM API to call
the existing agent functions as tools — listed as future work.

---

## 📚 Applications

- Disability-inclusive adaptive learning platforms
- IEP/504-integrated classroom tools
- Research into consent-respecting AI for vulnerable populations
- A template architecture for consent-first agentic AI beyond education

---

## ⚠️ Limitations

- All results are simulation-based; no real students, teachers, or IEP data were used
- OULAD provides only a binary disability flag, not sub-types; the six-subtype breakdown
  and all per-session behavioral dynamics are synthetic
- The Overload Prediction Agent's RandomForest is a documented, offline-feasible proxy for
  the GRU architecture originally specified
- "Explainability" (Teacher Co-Pilot Agent) measures rationale coverage, not human-judged
  understandability

Full discussion: [ARCHITECTURE.md](ARCHITECTURE.md)

---

## 📚 Citation

If referencing this work, please cite

```
Sharma, M. SCALE: A Consent-Aware Agentic AI Framework for Scaffolded,
Disability-Inclusive Education. 

---

## 👩‍💻 Author

**Mansi Sharma**
School of Computing, Indian Institute of Information Technology Una, Himachal Pradesh, India

Research interests:
- Agentic AI
- Disability-inclusive and accessible technology
- Reinforcement learning for education
- Consent-aware and explainable AI

---

## 📄 License

MIT License — see [`LICENSE`](LICENSE).
