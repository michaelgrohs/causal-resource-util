# Causal Analysis — Assumptions and Interpretation

## Shared assumptions (all methods)

**Backdoor criterion (no unobserved confounding)**
The covariate set X = {hour, dayofweek, n_alternatives, res_*} must block all
backdoor paths between T (resource utilisation) and Y (transition chosen).
Concretely: any variable that affects both util and the routing decision must be
captured in X. If a confounder is missing (e.g. case priority, queue length),
the estimates are biased.

**Stable Unit Treatment Value Assumption (SUTVA)**
The routing decision of one case does not affect the treatment or outcome of
another. Violated if, for example, one case occupying a resource directly
changes the util observed by a concurrent case in a way not captured by the
util value itself.

**Positivity (overlap)**
Every combination of covariate values X must have a non-zero probability of
receiving any treatment level T. If certain resources only ever appear at high
or low utilisation, effects for those resources cannot be identified from data.
The backdoor check's covariate balance (SMD) partially diagnoses this.

---

## `double_ml_test`

**What it estimates**
The partially linear regression coefficient θ in:

    Y − E[Y|X]  =  θ · (T − E[T|X])  +  ε

θ is the best-fit *linear* slope between residualised outcome and residualised
treatment.

**Additional assumptions**
- *Linearity*: the effect of util on P(chose k) is constant across all values of
  X and T. If the effect varies (heterogeneous CATEs), θ is a variance-weighted
  average of those varying effects — a valid summary, but not the simple mean.
- *Correct nuisance specification*: the Ridge models for E[Y|X] and E[T|X] must
  capture the relevant structure. Misspecification biases θ.

**Interpretation**
After partialling out X, a 1-unit increase in util changes P(chose k) by θ on
average. This equals an ATE only when the linearity assumption holds. Otherwise
it is a variance-weighted partial effect.

**Role in pipeline**
Fast significance screen. Use to decide whether util matters at all before
running the more expensive causal forest.

---

## `causal_forest_dml`

**What it estimates**
Per-sample Conditional Average Treatment Effects (CATEs) θ(Xᵢ), and their
average:

    ATE  =  mean_i [ θ(Xᵢ) ]

The CATE θ(Xᵢ) is the causal effect of a 1-unit increase in util on P(chose k)
for a case with covariates Xᵢ.

**Additional assumptions**
- *Honest splitting*: the forest uses separate subsamples for building tree
  structure and estimating leaf values. This yields valid CIs but halves the
  effective sample size — CIs are wider than double ML's.
- *Smooth heterogeneity*: the forest can recover heterogeneous effects, but
  assumes they vary smoothly enough to be captured by axis-aligned splits on X.
  Sharp discontinuities may be missed.

**Interpretation of ATE here vs. double ML**
The forest ATE is the mean of per-sample estimates, weighted uniformly over the
observed covariate distribution. The double ML θ is a variance-weighted slope.
They can diverge when the effect is nonlinear or when high-util cases cluster in
a particular subgroup.

**How to read the outputs together**

| Output | Question answered |
|---|---|
| ATE + CI | Is there an average effect? (less powerful than double ML for this) |
| CATE histogram | How much does the effect vary across cases? |
| CATE by resource | Which subgroup drives the effect? |
| CATE summary tree | Which covariate explains the heterogeneity? |
| Dose-response curve | Is the effect linear in util, or does it kick in at a threshold? |

A significant double ML θ with an insignificant forest ATE is not a
contradiction: the forest's variance estimator is more conservative. Trust
double ML for the binary "is there an effect?" question; trust the forest for
"who is affected and by how much?"

---

## `backdoor_check`

**Nuisance R²**
Cross-validated R² of predicting util from X. High R² means X strongly
predicts util — good, because those same variables then control for confounding.
R² near zero means util is nearly random given X, so confounding via X is
minimal (but unobserved confounders outside X are undetectable by this check).

**Placebo treatment test**
Replaces T with a random permutation and re-estimates θ. Should be near zero.
A large placebo θ suggests the observed effect is partly spurious — not
absorbed by X.

**Covariate balance (SMD)**
Standardised mean difference of each X between high-util and low-util groups.
|SMD| > 0.1 flags imbalance: the two groups differ on that covariate, which
could introduce bias if that covariate is also a confounder.

---

## What the analysis cannot establish

- **Causal direction**: the framework assumes util → choice. If workers
  anticipate which transition they will take and this affects their pace (and
  hence util), the direction is reversed.
- **External validity**: CATEs are estimated for the observed covariate
  distribution. Effects in unseen subgroups or under interventions that shift
  X are extrapolations.
- **Long-run effects**: the model is cross-sectional per event. Dynamic effects
  (e.g. high util today changing routing patterns tomorrow) are not captured.
