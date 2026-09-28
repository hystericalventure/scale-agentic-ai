# SCALE Architecture

This document describes how SCALE is structured and why, with the actual diagrams used
in the accompanying paper. For setup and usage, see the main [README](/README.md).

## The problem this addresses

Real data from the Open University Learning Analytics Dataset (OULAD, 32,593 students)
shows a measurable gap for disabled learners: a 38.1% vs. 48.2% pass/distinction rate, a
39.3% vs. 30.3% withdrawal rate, and materially lower platform engagement (1,327 vs. 1,539
mean clicks). SCALE is designed against this real, documented gap — not a hypothetical one.

## Design principles

SCALE is organized around three ideas that are new relative to prior multi-agent assistive
systems (most notably ADAPT, a health-domain agentic framework this project is architecturally
inspired by):

1. **Consent-first adaptation** — the system proposes changes and the student can accept,
   reject, or implicitly shape future proposals through a trust profile. Nothing is forced.
2. **Zone of Proximal Development (ZPD)-based scaffolding** — instead of adjusting a single
   difficulty number, the system reasons about *scaffolding depth*: independent work, a
   partial hint, or a full worked example.
3. **Proactive, not reactive, risk prediction** — the Overload Prediction Agent forecasts
   sensory/cognitive overload one session *ahead of time*, using only past behavioral trends,
   and triggers a preemptive break — rather than detecting a problem only after it happens.

## System architecture

Six parts once the persistent data store is counted: a multimodal interface, an LLM-based
decision layer for intent parsing, a Model Context Protocol (MCP) routing layer, the four
specialized agents, a Consent Ledger that reconciles agent outputs before anything reaches
the student or teacher, and an IEP/LMS/Behavioural Data Store that is not a processing
stage but the persistent record read by every layer and written only by the Consent Ledger.

![SCALE architecture](results/figures/diagram1_architecture.png)

*Fig. 1 (as in the paper): five processing layers on the left, the persistent data store as a tall box on the right. Dashed arrows are reads by every layer; the solid arrow is the Consent Ledger's write. The left-hand loop returns the consent-checked action to the learner. LaTeX (TikZ) source: `figures_tex/fig1_architecture.tex`.*

**Layer-by-layer (input → output):**

| Layer | Input | Output | Role |
|---|---|---|---|
| L1: Interface | student/teacher input | raw request | Multimodal input/output (voice, text, touch) |
| L2: LLM Interpretation | raw request | parsed intent | Parses free-text/voice intent *(designed, not implemented — see note below)* |
| L3: MCP Routing | parsed intent | routed request | Dispatches the request to the agent(s) responsible for it |
| L4: Multi-Agent Layer | routed request + learner state | 4 candidate actions | The four agents described below, each proposing one candidate action |
| L5: Consent Ledger | 4 candidate actions | 1 consent-checked action | Enforces priority: safety/IEP compliance > student consent > pedagogy > nudges |
| Data Store | — | persistent record | IEP/LMS/behavioural data, read by every layer (dashed arrows), written only by the Consent Ledger (solid arrow) |

> **Implementation note:** what is actually built and evaluated in this repository is a
> simplified version of this architecture. The four agents (L4) are built and trained as
> specified, with two forced substitutions (a simulated consent signal in place of real
> learners, and a RandomForest in place of the originally-specified GRU for overload
> prediction). L2 (the LLM layer) is not built at all — the agents read structured numeric
> signals directly. L1, L3, L5, and the data store are each implemented in a reduced form
> (synthetic learners emitting numeric signals instead of live multimodal input; direct,
> fixed-order function calls instead of an MCP server; a fixed per-session consent/trust
> update instead of a logged, persistent ledger; a CSV of 400 synthetic profiles instead of
> a live LMS/IEP integration). None of these substitutions change what the four agents do —
> see the main README's "Architecture vs. implementation" section, and Table 1 of the paper,
> for the full per-layer breakdown and rationale.

## The four agents

### 1. ZPD Scaffolding Agent
Tabular **Q-learning** over a 45-state space (mastery level × fatigue level × recent error
rate), choosing between three actions: independent work, a partial hint, or a full worked
example. Learns purely from trial-and-error interaction with the simulated environment —
it never has access to the "correct" scaffold level, only behavioral feedback.

### 2. Consent & Trust Agent
Every scaffold change proposed by the ZPD agent is gated through this agent before
execution. Acceptance probability depends on current trust and how large the proposed
change is; rejecting a suggestion still updates trust (respecting a "no" is informative,
not wasted). If rejected, the system falls back to the student's current level rather than
forcing the change.

### 3. Overload Prediction Agent
A RandomForest classifier trained to forecast overload **one session ahead**, using only
the trend of behavioral signals (response latency, error rate, hesitation, fatigue) from
the two preceding sessions — never the current session's own data. This is a genuinely
harder task than detecting an anomaly after it's already visible, and is evaluated as such.

### 4. Teacher Co-Pilot Agent
Generates a plain-language, data-grounded rationale for every decision, and flags students
whose trust, acceptance rate, or overload trend suggests they need teacher attention. Flags
are checked against actual outcomes to confirm they're informative, not just noise.

## The Consent-Aware PRAC loop

Each agent runs an extended Perception-Reasoning-Action loop with an added **Consent**
stage — the core mechanism that distinguishes SCALE's coordination model from a plain
Perception-Reasoning-Action loop.

![Consent-Aware PRAC loop](results/figures/diagram2_prac_loop.png)

*Fig. 2 (as in the paper): the five-stage loop each agent runs. LaTeX (TikZ) source: `figures_tex/fig2_prac_loop.tex`.*

No action executes on the student without either explicit consent or a pre-authorized
safety override. Every outcome — accepted or rejected — updates the trust model, which
shapes how future proposals are made.

## Results at a glance

Full experimental methodology and figures are in [RESULTS.md](RESULTS.md).
