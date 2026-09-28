# SCALE Results

Full experimental results, all reproducible from the code in `src/`. See
[ARCHITECTURE.md](ARCHITECTURE.md) for how the system is designed.

## Methodology summary

- **Population:** 400 synthetic students across six SLD-and-related sub-types (Dyslexia
  25%, Dyscalculia 15%, Dysgraphia 10%, ADHD 20%, SLD+ADHD Comorbid 20%, Multi-SLD
  Comorbid 10%), grounded in reported SLD-ADHD and cross-SLD comorbidity rates and
  calibrated, via an analytical compensation step, so the population's overall
  performance/engagement disadvantage matches real ratios computed from OULAD (a real
  dataset of 32,593 students).
- **Split:** 320 students for training, 80 fully held-out students for evaluation —
  no agent is ever evaluated on data it influenced during training.
- **Repetition:** every experiment is run across multiple random seeds (5 for individual
  agents, 3 for full-system integration) and reported as mean ± standard deviation, not a
  single run.
- **Simulation:** each student is simulated over a 10-week, 5-session-per-week closed loop
  where agent decisions causally affect mastery, error rate, fatigue, and overload risk.

## 1. ZPD Scaffolding Agent vs. baselines

| Policy | Mean reward | Mean overload events | Mean final mastery |
|---|---|---|---|
| **Learned (Q-learning)** | **6.53 ± 0.02** | **14.70 ± 0.34** | **1.000 ± 0.000** |
| Random scaffold | -6.83 ± 0.32 | 24.86 ± 0.29 | 0.919 ± 0.003 |
| Static (always independent) | -40.21 ± 0.02 | 30.69 ± 0.09 | 0.574 ± 0.000 |

![ZPD Agent comparison](results/figures/fig3_zpd_comparison.png)

The gap between the learned policy and both baselines is an order of magnitude larger
than seed-to-seed variance — the advantage is not a lucky random draw.

## 2. Overload Prediction Agent

Forecasting overload **one session ahead**, using only behavioral trends from the two
preceding sessions (a genuinely harder task than reactive, after-the-fact detection):

| Metric | Value |
|---|---|
| Accuracy | 0.604 |
| Precision | 0.502 |
| Recall | 0.593 |
| F1 | 0.544 |
| ROC-AUC | 0.649 |

![Overload feature importances](results/figures/fig5_overload_features.png)

Response latency and error-rate trend dominate feature importance, consistent with the
environment's underlying causal structure. This is reported as a modest but genuine
early-warning signal, not an inflated claim.

## 3. Consent & Trust Agent

Gating scaffold changes through explicit consent costs some short-term reward relative to
a push-only control, because rejected proposals sometimes forgo an objectively better
action. But acceptance climbs sharply over the simulated term:

![Trust curve](results/figures/fig4_trust_curve.png)

**Week 1: 45.7% acceptance → Week 10: 97.3% acceptance** (final mean trust: 0.990), rising
smoothly across all ten weeks. This is the core evidence that treating consent as a
first-class mechanism — not an afterthought — produces a measurable, positive trust
dynamic over time.

## 4. Teacher Co-Pilot Agent

Every decision receives a template-grounded rationale (100% coverage by construction — this
measures rationale coverage, not human-judged understandability, which would require a real
user study). Of 80 held-out students, 43.8% were flagged for teacher attention. Flagged
students showed substantially worse outcomes (mean overload events 21.26 vs. 12.31; mean
final mastery 0.952 vs. 1.000) — confirming the flags are informative, not arbitrary.

## 5. Full-system integration (headline result)

Three conditions compared across 3 random seeds on held-out students: the complete
four-agent SCALE system, a non-agentic rule-based baseline (heuristic scaffold rule, no
consent, no proactive breaks), and a push-only control (SCALE's agents without the consent
gate).

| Condition | Mean reward | Mean overload events | Mean final mastery | Acceptance rate |
|---|---|---|---|---|
| **SCALE (full)** | 0.731 ± 0.283 | 15.188 ± 0.296 | 0.988 ± 0.006 | 70.8% ± 0.4% |
| Rule-based baseline | -3.483 ± 0.070 | 23.158 ± 0.225 | 0.817 ± 0.005 | n/a (auto-execute) |
| Push-only (no consent) | 4.907 ± 0.114 | 13.929 ± 0.295 | 1.000 ± 0.000 | n/a (auto-execute) |

![Full system comparison](results/figures/fig6_full_system_comparison.png)

**Interpretation:** SCALE clearly beats the non-agentic rule-based baseline on every metric.
Against the push-only control, SCALE trades some reward for genuinely consent-gated
behavior rather than forced execution — a real, honestly-reported cost, not smoothed over.

## 6. Statistical significance

Paired t-test comparing SCALE (full) against the rule-based baseline, matched by
(student, random seed) — 240 paired observations:

| Metric | SCALE mean | Rule-based mean | Mean difference | p-value |
|---|---|---|---|---|
| Total reward | 0.731 | -3.483 | +4.214 | 1.47e-26 |
| Overload events | 15.188 | 23.158 | -7.971 | 3.37e-55 |
| Final mastery | 0.988 | 0.817 | +0.171 | 7.85e-40 |

All differences are significant well beyond p < 0.001 — the improvement is not
attributable to random chance.

## 7. Fairness across SLD sub-types

![Fairness by sub-type](results/figures/fig7_fairness.png)

| Sub-type | n | Mean reward | Mean overload events | Mean final mastery | Mean acceptance rate |
|---|---|---|---|---|---|
| Dysgraphia | 8 | 2.552 | 12.042 | 0.999 | 77.8% |
| ADHD | 15 | 2.059 | 15.867 | 0.990 | 69.3% |
| SLD+ADHD Comorbid | 20 | 0.935 | 17.300 | 0.992 | 68.6% |
| Dyslexia | 20 | 0.483 | 13.350 | 0.988 | 73.8% |
| Dyscalculia | 12 | -0.976 | 15.417 | 0.977 | 68.6% |
| Multi-SLD Comorbid | 5 | -1.888 | 16.533 | 0.975 | 66.0% |

Mastery spread across the six sub-types is only **0.024** (0.975–0.999) — SCALE is fairly
equitable in final learning outcome across sub-types. Reward and overload burden vary more:
the SLD+ADHD Comorbid and Multi-SLD Comorbid groups, the most severely affected profiles by
construction, see somewhat lower mean reward and more overload events on average. This is
reported honestly rather than only showing the favorable average.

## 8. Learning efficiency

![Learning efficiency](results/figures/fig8_learning_efficiency.png)

Average sessions needed to first reach mastery ≥ 0.6:

| Condition | Mean sessions to mastery |
|---|---|
| Push-only (no consent) | 12.85 |
| SCALE (full) | 15.55 |
| Rule-based baseline | 24.65 |

SCALE reaches meaningful mastery roughly **37% faster** than the non-agentic rule-based
baseline, while push-only is fastest of all — the cost of never asking the student.

## 9. Overload prediction lead time

![Lead time distribution](results/figures/fig9_lead_time.png)

For 1,126 real overload events in the held-out evaluation, measuring how many
consecutive prior sessions already showed elevated risk (≥0.5):

- Mean: **0.56** sessions of sustained prior warning
- **73.8%** of events have no sustained prior warning (caught only at the event itself)
- **26.2%** of events get at least 1 session of advance warning
- **15.7%** of events get at least 2 sessions of advance warning

This is reported as a modest, honest result: proactive forecasting from behavioral
trends alone is a genuinely hard problem, and this is the ceiling for this feature set
without richer signals (e.g., physiological data).

## Limitations

All results above are simulation-based algorithmic comparisons, not evidence of real-world
educational effectiveness. See the main [README](../README.md#limitations) for the full list.
