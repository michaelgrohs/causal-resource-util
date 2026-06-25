# Causal Analysis of Inter-Case Effects in Business Process Execution Logs

## Overview

This document describes the complete analytical pipeline used to investigate whether **resource utilisation** causally influences **routing decisions** at non-deterministic choice points in business process execution logs. The pipeline takes Petri net execution traces as input and produces, for each identifiable decision point in the process, a suite of causal estimates ranging from a simple debiased linear coefficient to heterogeneous per-case treatment effects and dose-response curves. Each analytical step is accompanied by diagnostic tests designed to assess the plausibility of the causal identification assumptions.

---

## 1. Data Representation and Preprocessing

### 1.1 Input Format: SLPN Execution Files

**What.** The raw input to the pipeline consists of execution trace files in the `.exs` format, produced by the EBI process mining toolkit from Stochastic Labelled Petri Nets (SLPNs) discovered from event logs. Each file encodes a sequence of transition firings across all cases (process instances) in the log, together with information about which other transitions were enabled at the moment of each firing.

**Why.** Standard event logs record *what happened* but not *what could have happened instead*. The `.exs` format closes this gap: by recording the full set of enabled transitions at each firing, it makes the available routing alternatives explicit. This is essential for causal analysis, because a causal effect can only be defined relative to a comparison — which alternative path was not taken — and that comparison must be grounded in the actual process structure, not a post-hoc statistical approximation.

**How.** Each entry in an `.exs` file specifies: the case identifier (`trace`), the activity label of the fired transition (`activity`, or `null` for silent/invisible transitions), the integer identifier of the fired transition, the list of other enabled transition identifiers at the time of firing (`other_enabled_transitions`), the executing resource and its utilisation at the moment of firing (`resource`, `resource_utilisation`), and the execution timestamp. The parser recovers all of these fields, normalises timestamps to UTC, enforces type consistency (integer transition ids, float utilisation in $[0,1]$, timezone-aware datetimes), and assembles them into a tidy DataFrame with one row per firing event.

### 1.2 Filtering to Genuine Decision Points

**What.** From the full set of firings, we retain only those rows that correspond to *genuine routing decisions*: visible (non-silent) transitions that fire at a point in the Petri net where at least one other visible alternative was simultaneously enabled.

**Why.** The causal question — does resource utilisation influence *which route is taken* — is only meaningful at XOR-type decision points, where the process had a real fork. Silent transitions are structural placeholders with no observable activity label and no resource. Single-enabled firings are not decisions at all: no alternative exists, so there is nothing to explain. Restricting the analysis to genuine decision points ensures that the choice is conceptually contestable and that the counterfactual ("what if a different transition had fired?") is grounded in the process structure.

**How.** A row is classified as a decision point when (a) `activity` is non-null (visible transition), (b) `resource_utilisation` is a valid numeric value, and (c) the `other_enabled` list is non-empty. Silent transitions and deterministic firings are excluded. For each retained row, the *choice set* is constructed as the sorted tuple of activity labels of all simultaneously enabled visible transitions, including the one that actually fired.

### 1.3 Defining the Choice Set and Samples

**What.** Each decision point is assigned to a *choice set* — the tuple of activity labels that were concurrently available at that moment. All statistical analyses are performed independently within each choice set, treating each such set as a separate experimental context.

**Why.** Choice sets partition the data into structurally homogeneous subpopulations. A decision between transitions $\{A, B\}$ is fundamentally different from a decision between $\{A, B, C\}$: the baseline probability of each route, the confounding structure, and the counterfactual comparison all differ. Pooling across choice sets would conflate these distinct decision contexts and produce misleading effect estimates. Stratifying by choice set ensures that every comparison is like-for-like: each sample in a given stratum faced exactly the same menu of options.

**How.** The `choice_set` column is formed by taking the union of the fired transition label and all labels in `other_enabled`, sorting alphabetically, and converting to a tuple. All downstream functions use `decisions.groupby("choice_set")` so that models are fitted and effects are estimated separately for each stratum.

---

## 2. Variables: Treatment, Outcome, and Confounders

### 2.1 Treatment: Resource Utilisation

**What.** The treatment variable $T$ is `resource_utilisation`, a continuous value in $[0, 1]$ representing how occupied the executing resource was at the moment the transition fired. A value of $0$ indicates an idle resource; a value of $1$ indicates full occupancy.

**Why.** Resource utilisation is the primary *inter-case effect* variable: it encodes the influence of other concurrently running cases on the current case's context. When a resource is heavily loaded (high utilisation), it may — either through deliberate routing logic or through implicit behavioural adaptation — favour certain transition paths over others. Establishing whether this influence is causal, rather than merely associative, is the central research question.

**How.** The value is taken directly from the `.exs` execution record. It is used as a *continuous* treatment throughout the causal analyses (Double ML and Causal Forest DML), preserving the full dose-response structure without information loss. A binary threshold variable (`treatment = 1` if utilisation $\geq 0.5$) is constructed separately for diagnostic purposes (covariate balance checks) but is not used as the primary treatment variable in either causal estimator.

### 2.2 Outcome: Transition Choice (One-vs-Rest Binary)

**What.** For each transition $k$ in a choice set, the outcome variable is $Y_k = \mathbf{1}[\text{chosen} = k]$, a binary indicator of whether transition $k$ was the one that fired. Each transition within a choice set gives rise to a separate outcome variable, and the causal analysis is repeated independently for each $k$.

**Why.** A one-vs-rest encoding makes the outcome binary without imposing a multinomial model or a shared parametric structure across transitions. This is appropriate here because the transitions may differ qualitatively (e.g. approval vs. rejection, fast-track vs. standard route) and there is no natural ordering among them. The one-vs-rest formulation also allows each transition's effect to be estimated with a different signal-to-noise ratio, and enables the identification of which specific path is systematically favoured or disfavoured under high utilisation.

**How.** For each transition $k$ encountered in a choice set group, the binary outcome is computed as `Y = (chosen == k).astype(float)`. The causal estimators are then applied to $(Y_k, T, X)$ for each $k$ independently. Because $\sum_k Y_k = 1$ for every row, the effects across transitions within a choice set will sum to approximately zero (a unit increase in $T$ that raises the probability of one path must correspondingly lower it for the others), providing a natural consistency check.

### 2.3 Covariates (Confounding Variables)

**What.** The covariate matrix $X$ consists of variables that are believed to confound the relationship between utilisation and routing choice. The following features are used:

| Feature | Derivation | Role |
|---|---|---|
| `hour` | Hour of execution timestamp | Captures intraday workload patterns |
| `dayofweek` | Day of week of execution timestamp | Captures weekly rhythms in routing behaviour |
| `n_alternatives` | Size of the choice set at this decision | Controls for structural differences in how many routes are available |
| `res_<name>` | One-hot encoding of the executing resource | Captures resource-specific routing preferences and workload profiles |

**Why.** Confounders are variables that causally affect *both* the treatment (utilisation) and the outcome (which route is chosen), thereby creating a spurious association between the two. Each covariate in $X$ addresses a plausible confounding pathway:

- **Temporal features** (`hour`, `dayofweek`) reflect the time of day and week, which jointly determines both how busy resources tend to be and how process routing tends to unfold. For instance, early morning hours may simultaneously see lower utilisation and a higher frequency of standard-path routing.
- **Number of alternatives** (`n_alternatives`) captures the structural context of the decision: more alternatives may correlate with lower utilisation (early stages of complex processes) or may independently influence which transitions are chosen.
- **Resource identity** (`res_<name>`) is the single most important confounder: different resources operate at characteristically different utilisation levels (reflecting different workload assignments) and may also have different routing preferences or authority levels. Without controlling for resource identity, any resource-level difference in both utilisation and routing behaviour would appear as a causal effect of utilisation.

**How.** Temporal features are extracted from the `time_of_execution` timestamp using pandas datetime accessors. Resource identity is one-hot encoded via `pd.get_dummies`, producing a binary column `res_<name>` for each distinct resource observed in the dataset. This avoids imposing an arbitrary ordinal structure on what is a nominal variable. The full feature list for a given dataset is retrieved via `get_feature_cols(decisions)`, which returns the base temporal features concatenated with all `res_*` columns present in that dataset.

### 2.4 On Binarisation of the Treatment

**What.** Binarisation of the treatment refers to discretising the continuous utilisation variable into two groups (e.g. "high" vs. "low") using a threshold, typically $T_{\text{bin}} = \mathbf{1}[T \geq 0.5]$.

**Why this is not done as the primary analysis.** Binarisation discards information and introduces threshold-dependence. The threshold $0.5$ is arbitrary: whether the true dose-response relationship is monotone, threshold-triggered, or non-monotone cannot be determined from a binary split. More critically, the causal effect of a continuous treatment is, in general, a function of the dose, not a single number. The Double ML and Causal Forest DML estimators are both designed for continuous treatments and preserve the full dose-response structure. Forcing a binary treatment would collapse this structure unnecessarily.

**When binarisation is used.** A binary variable `treatment` (utilisation $\geq 0.5$) is retained solely for the covariate balance check in the backdoor adequacy tests. Standardised mean differences (SMDs) between the high- and low-utilisation groups serve as a diagnostic for whether the covariates $X$ are sufficiently balanced across treatment levels. This diagnostic use is appropriate because SMD is defined for two groups; it does not imply that the causal analysis itself uses a binary treatment.

---

## 3. Step 1 — Association Screening: Logistic Regression with Permutation Test

**What.** Before applying computationally intensive causal estimators, we conduct a conditional association test for each choice set. The test asks: does the probability distribution over choices within a given choice set depend on utilisation, after marginalising out other structure?

**Why.** The permutation test provides a fast, assumption-light null-hypothesis test of the form $H_0: \text{choice} \perp T \mid \text{choice\_set}$. A significant association is a necessary (but not sufficient) condition for a causal effect. Conversely, a null result across all transitions in a choice set is strong evidence against a meaningful causal effect and warrants stopping the analysis early. The test is also robust to class imbalance and does not require correct specification of the effect size.

**How.** For each choice set, a multinomial logistic regression is fitted with `util` as the sole predictor and the chosen transition label as the outcome. The observed coefficient on `util` is compared against a null distribution obtained by refitting the model on 2,000 permutations of the outcome labels (keeping `util` fixed). The permutation p-value is computed as the fraction of null coefficients whose absolute value exceeds the observed coefficient. A 95% bootstrap confidence interval is also reported for the observed coefficient.

---

## 4. Step 2 — Double ML: Debiased Linear Estimate

### 4.1 Goal

The Double ML (Debiased Machine Learning) estimator, introduced by Chernozhukov et al. (2018), addresses the fundamental challenge of causal inference from observational data in the presence of high-dimensional confounders. Its goal is to estimate the average *linear* causal effect of utilisation on the probability of choosing a given transition, controlling for the full covariate set $X$ in a way that is robust to model misspecification and regularisation bias.

### 4.2 What It Computes

Double ML fits the **Partially Linear Regression (PLR)** model:

$$Y_k - \mathbb{E}[Y_k \mid X] = \theta_k \cdot (T - \mathbb{E}[T \mid X]) + \varepsilon$$

The parameter $\theta_k$ is the constant causal slope: a one-unit increase in resource utilisation changes the probability of choosing transition $k$ by $\theta_k$, on average, holding all covariates constant. Because the model is linear in $T$, $\theta_k$ is a single number — a *global* linear effect. Under the linearity assumption, $\theta_k$ equals the Average Treatment Effect (ATE). When the true effect is heterogeneous, $\theta_k$ recovers a variance-weighted average of the heterogeneous effects, which remains a valid and interpretable summary statistic (Chernozhukov et al., 2018).

### 4.3 How It Computes It

The key insight of Double ML is to *residualise* both the treatment and the outcome with respect to the covariates before estimating the causal coefficient. This two-step procedure breaks the spurious association between $T$ and $Y_k$ that is attributable to $X$, leaving only the variation in $T$ that is *unexplained* by the covariates to identify $\theta_k$.

**Step 1 — Residualise the treatment.** A nuisance model $\hat{\mathbb{E}}[T \mid X]$ is fitted using cross-fitted Ridge regression with 5-fold cross-validation. The treatment residual is:
$$\tilde{T}_i = T_i - \hat{\mathbb{E}}[T_i \mid X_i]$$
This residual captures variation in utilisation that is *not* explained by the observed context, approximating what randomisation would achieve experimentally.

**Step 2 — Residualise the outcome.** For each transition $k$, a nuisance model $\hat{\mathbb{E}}[Y_k \mid X]$ is fitted using the same cross-fitted Ridge regression. The outcome residual is:
$$\tilde{Y}_{k,i} = Y_{k,i} - \hat{\mathbb{E}}[Y_{k,i} \mid X_i]$$

**Step 3 — Estimate $\theta_k$ by OLS on residuals.** The causal coefficient is obtained by regressing $\tilde{Y}_k$ on $\tilde{T}$:
$$\hat{\theta}_k = \frac{\tilde{T}^\top \tilde{Y}_k}{\tilde{T}^\top \tilde{T}}$$

**Step 4 — Compute HC3-robust standard errors.** To obtain valid inference without assuming homoscedastic errors, heteroscedasticity-consistent (HC3) standard errors are computed using the leave-one-out leverage adjustment, which accounts for the influence of individual observations on the estimate. A 95% confidence interval and two-sided Wald test are reported.

**Cross-fitting** is critical: by fitting each nuisance model on a held-out fold and predicting on the complementary fold, cross-fitting eliminates the regularisation bias that would arise if the same data were used for nuisance estimation and effect identification. This ensures that the final estimate is $\sqrt{n}$-consistent even when the nuisance models are not (Chernozhukov et al., 2018).

### 4.4 Assumptions

Beyond the shared identification assumptions (backdoor criterion, SUTVA, positivity — see Section 9), Double ML requires:

- **Linearity.** The effect of $T$ on $Y_k$ is the same for all values of $X$ and $T$. If the effect varies across subgroups (heterogeneity), $\hat{\theta}_k$ recovers a variance-weighted average rather than a simple mean.
- **Correct nuisance specification.** The Ridge models for $\mathbb{E}[Y_k \mid X]$ and $\mathbb{E}[T \mid X]$ must capture enough of the conditional expectation to avoid residual confounding. Cross-fitting mitigates but does not eliminate this concern.

---

## 5. Step 3 — Causal Forest DML

### 5.1 Goal

The Causal Forest DML estimator — combining the Causal Forest of Wager and Athey (2018) with the Double ML nuisance residualisation of Chernozhukov et al. (2018), as implemented by Athey et al. (2019) in the `econml` library — relaxes the linearity assumption of Step 2. Its goal is to estimate *heterogeneous* causal effects: the extent to which the effect of utilisation on routing choice varies across cases with different characteristics, and to characterise which characteristics drive that variation.

### 5.2 What It Computes

The Causal Forest DML estimates the **Conditional Average Treatment Effect (CATE)**:

$$\theta(X_i) = \mathbb{E}[Y_k(T+1) - Y_k(T) \mid X = X_i]$$

where $Y_k(t)$ is the potential outcome (probability of choosing transition $k$) under utilisation level $t$. The CATE $\theta(X_i)$ is a function of $X$ — a different number for each combination of covariate values — rather than a single constant. The **Average Treatment Effect** (ATE) is recovered as the empirical mean of the per-case CATEs:

$$\widehat{\text{ATE}}_k = \frac{1}{n}\sum_{i=1}^n \hat{\theta}(X_i)$$

This ATE is the uniformly-weighted mean over the observed covariate distribution, in contrast to the variance-weighted Double ML estimate from Step 2.

In addition to point estimates, the forest produces per-sample 95% confidence intervals via the infinitesimal jackknife variance estimator (Wager et al., 2014), enabling valid inference on both the ATE and on individual CATEs.

### 5.3 How It Computes It

**Step 1 — Nuisance residualisation (Double ML stage).** Gradient-boosted regression trees are fitted cross-fittedly (5-fold) to estimate both $\hat{\mathbb{E}}[T \mid X]$ and $\hat{\mathbb{E}}[Y_k \mid X]$. The residuals $\tilde{T}$ and $\tilde{Y}_k$ are formed as in Step 2. Gradient boosting is preferred over Ridge at this stage because the conditional expectations $\mathbb{E}[T \mid X]$ and $\mathbb{E}[Y_k \mid X]$ may be non-linear in $X$, and non-parametric nuisance estimation increases the robustness of the downstream CATE estimates.

**Step 2 — Causal Forest on residuals.** A random forest of causal trees is grown on the residuals, targeting:
$$\tilde{Y}_{k,i} \approx \theta(X_i) \cdot \tilde{T}_i$$
Each tree is built using the **generalised Robinson decomposition** (Robinson, 1988): the criterion for splitting is based on minimising the within-leaf heterogeneity of the moment condition $(\tilde{Y}_k - \theta \cdot \tilde{T})^2$, not a standard regression criterion on $Y$. This ensures that the splits isolate regions of $X$ where the causal effect genuinely differs, rather than regions where the conditional mean of $Y$ differs.

**Honest splitting** (Athey and Imbens, 2016) is applied: each tree uses one subsample to determine the split structure and a disjoint subsample to estimate the leaf-level CATE. This two-sample discipline ensures that the leaf estimates are not over-fit to the data used to construct the splits, which is a necessary condition for valid confidence intervals.

The forest produces, for each observation $i$: a point estimate $\hat{\theta}(X_i)$, and a 95% confidence interval $[\hat{\theta}^{\text{lb}}(X_i), \hat{\theta}^{\text{ub}}(X_i)]$ via the infinitesimal jackknife.

**Step 3 — Interpretable summary tree.** To identify which covariates drive heterogeneity, a shallow (depth-3) CART decision tree is fitted on the per-sample CATE estimates $\hat{\theta}(X_i)$ with the feature matrix $X$ as predictors. This summary tree partitions the covariate space into regions with similar CATEs, making the heterogeneity structure interpretable. Feature importances from this tree indicate which variables most strongly moderate the causal effect of utilisation.

### 5.4 Assumptions

Beyond the shared identification assumptions, the Causal Forest DML requires:

- **Honest splitting.** The split-subsample and estimation-subsample must be statistically independent. This is guaranteed by construction but halves the effective sample size per tree, leading to wider confidence intervals than Double ML.
- **Smooth heterogeneity.** The CATE function $\theta(X)$ is assumed to vary smoothly enough to be approximated by a piecewise-constant function over axis-aligned splits on $X$. Sharp discontinuities in $\theta(X)$ may be missed or smoothed over by the forest.
- **Sufficient sample size.** Each leaf must contain at least `min_samples_leaf = 10` observations (by default) to produce stable estimates. Choice sets with fewer than 30 total observations are excluded.

---

## 6. Comparison: Double ML versus Causal Forest DML

| Dimension | Double ML (Step 2) | Causal Forest DML (Step 3) |
|---|---|---|
| **Model for effect** | Linear: $\theta$ constant across all $X$, $T$ | Non-parametric: $\theta(X)$ varies freely with $X$ |
| **Output** | Single coefficient $\hat{\theta}_k$ per transition | Per-case CATE $\hat{\theta}(X_i)$ + ATE |
| **Nuisance models** | Ridge regression | Gradient Boosting |
| **Standard errors** | HC3-robust, analytical | Infinitesimal jackknife, from forest |
| **Confidence intervals** | Narrower (parametric) | Wider (honest splitting) |
| **Best for** | Fast significance screen: "Is there an effect?" | Characterising who is affected and how much |
| **Heterogeneity** | Masks it (variance-weighted average) | Explicitly models it |
| **Dose shape** | Assumes linear (single slope per unit $T$) | No assumption (CADR recovers the shape) |
| **Key assumption** | Linearity of $\theta$ in $T$ | Smooth variation of $\theta(X)$ across $X$ |
| **Sample requirement** | $n \geq 2 \times$ number of folds | $n \geq 30$; honest splitting halves effective $n$ |

**Practical guidance.** A significant Double ML $\hat{\theta}_k$ with a non-significant Forest ATE is not a contradiction: the forest's confidence intervals are more conservative by design. The two methods should be read in tandem: Double ML answers "does utilisation matter at all for this transition?" while the Causal Forest answers "for whom does it matter, and by how much?" If the Double ML estimate is near zero and non-significant, the Forest results should be interpreted with caution regardless of their point estimates.

---

## 7. Effects Generated by the Pipeline

The pipeline produces four distinct effect quantities, each answering a different causal question:

| Effect | Symbol | Method | Interpretation |
|---|---|---|---|
| **Linear ATE** | $\hat{\theta}_k$ | Double ML | Average change in $P(\text{chose } k)$ per unit increase in utilisation, assuming linearity |
| **Forest ATE** | $\widehat{\text{ATE}}_k$ | Causal Forest DML | Uniformly-weighted average of per-case CATEs; same interpretation as above but without linearity assumption |
| **CATE** | $\hat{\theta}(X_i)$ | Causal Forest DML | Case-specific causal effect: how much does a unit increase in utilisation change $P(\text{chose } k)$ for a case with covariates $X_i$? |
| **Dose response** | $\text{CADR}(t)$ | Conditional Average Dose Response | Average causal effect of *setting* utilisation to $t$ versus setting it to $0$, as a function of $t$ |

These effects are estimated separately for each transition $k$ within each choice set, yielding a full causal profile of how utilisation shapes process routing at each decision point. The signs of the effects across transitions within a choice set provide a coherent picture: a positive effect for transition $k$ implies a corresponding negative aggregate effect for the remaining transitions (since probabilities must sum to one).

---

## 8. Step 4 — Conditional Average Dose Response (CADR)

### 8.1 Goal

The CADR addresses a limitation of both the Double ML and the Forest ATE: both produce a single scalar summary of the effect of a one-unit increase in $T$. If the dose-response relationship is non-linear — for instance, if utilisation has no effect below $0.3$ but a sharp effect above $0.8$ — a single linear coefficient or an average effect at a single contrast will not reveal this. The CADR instead traces the full functional relationship between $T$ and the causal effect.

### 8.2 What It Computes

For a grid of treatment levels $\{t_1, t_2, \ldots, t_m\} \subset [0, 1]$, the CADR estimates:

$$\text{CADR}(t) = \mathbb{E}_X\left[\mathbb{E}[Y_k(t) - Y_k(t_{\text{ref}}) \mid X]\right]$$

where $t_{\text{ref}} = 0$ is the baseline (idle resource). For each $t$ on the grid, this quantity answers: "on average across all cases (with their observed covariates), how does setting utilisation to $t$ change the probability of choosing transition $k$ relative to an idle resource?" The curve $\text{CADR}(\cdot)$ reveals whether the effect is linear, threshold-triggered, saturating, or non-monotone.

### 8.3 How It Computes It

For each fitted Causal Forest model (one per transition per choice set) and each $t$ on the grid, the `effect(X, T0=t_ref, T1=t)` method of the fitted `CausalForestDML` object is called on the full covariate matrix $X$. This returns a per-sample estimate of the causal effect of moving from $t_{\text{ref}}$ to $t$ for each observation. The mean across observations and the mean of the per-sample 95% confidence bounds are reported. This is repeated for each $t$ in the grid (default: 30 evenly-spaced points in $[0, 1]$), producing a curve with pointwise confidence bands.

---

## 9. Step 5 — Backdoor Adequacy Tests

### 9.1 Goal

The causal interpretation of all estimates in this pipeline rests on the **backdoor criterion** (Pearl, 2009): the covariate set $X$ must block all confounding paths between $T$ (utilisation) and $Y_k$ (routing choice). Since this is an observational dataset — utilisation is not experimentally assigned — the criterion cannot be verified from data alone. The backdoor adequacy tests instead provide three complementary diagnostic checks that make the plausibility of this assumption explicit and falsifiable.

### 9.2 Check 1 — Nuisance $R^2$ (Treatment Predictability)

**What.** Cross-validated $R^2$ of predicting $T$ from $X$ using Ridge regression.

**Why.** If $X$ explains a large fraction of the variance in $T$, the same variables that drive utilisation differences are controlled for in the analysis, reducing the scope for unobserved confounding *via* the observed pathways. Conversely, if $R^2 \approx 0$, then $X$ barely explains utilisation: either utilisation is (near-)random conditional on $X$ (little confounding via these variables), or the confounders are missing from $X$ entirely. A very low $R^2$ is not alarming by itself but should be interpreted together with Check 2.

**How.** Ridge regression is cross-fitted with 5 folds. $R^2$ is computed from the cross-validated predictions. Values above $0.1$ are flagged as indicating meaningful $X$-to-$T$ predictability.

### 9.3 Check 2 — Placebo Treatment Test

**What.** A random permutation of $T$ is used as a "placebo treatment" in the Double ML estimator. The resulting coefficient should be approximately zero if $X$ successfully absorbs all confounders.

**Why.** Under the null hypothesis that $X$ fully blocks confounding, the residuals $\tilde{Y}_k = Y_k - \hat{\mathbb{E}}[Y_k \mid X]$ should be orthogonal to any version of $T$ that is statistically independent of the true effect. Replacing $T$ with a random permutation $T_\pi$ (which by construction has no causal effect on $Y_k$) and re-running the Double ML procedure should yield $\hat{\theta}^\pi_k \approx 0$. A systematically large $\hat{\theta}^\pi_k$ across permutations indicates that $X$ has not absorbed a confounding signal that correlates both with $T$ and $Y_k$ — suggesting an uncontrolled confounder (Zhao and Hastie, 2021).

**How.** 200 independent random permutations of $T$ are generated. For each, the Double ML procedure is run and the placebo coefficient is recorded. The mean and standard deviation of these placebo coefficients are reported. A mean placebo coefficient exceeding twice its standard deviation is flagged.

### 9.4 Check 3 — Covariate Balance (Standardised Mean Difference)

**What.** The standardised mean difference (SMD) of each covariate $X_j$ between the high-utilisation group ($T \geq$ median) and the low-utilisation group ($T <$ median).

**Why.** In a randomised experiment, treatment assignment is independent of covariates: the two groups have the same distribution of $X$ in expectation, i.e. SMD $\approx 0$ for all covariates. In an observational study, large SMDs indicate that the two groups differ systematically on covariates that may also influence the outcome — the signature of confounding (Austin, 2011). If a covariate $X_j$ both strongly predicts $T$ (large SMD) and strongly predicts $Y_k$, then it is a confounder that must be controlled for. This check verifies whether the observed covariates exhibit the expected imbalance and thus whether controlling for them is materially important.

**How.** The SMD for covariate $X_j$ is computed as:
$$\text{SMD}_j = \frac{\bar{X}_j^{\text{hi}} - \bar{X}_j^{\text{lo}}}{\sqrt{(\hat{\sigma}_j^{\text{hi}\,2} + \hat{\sigma}_j^{\text{lo}\,2}) / 2}}$$
where $\bar{X}_j^{\text{hi/lo}}$ and $\hat{\sigma}_j^{\text{hi/lo}}$ are the group means and standard deviations. The conventional threshold of $|\text{SMD}| > 0.1$ is used to flag imbalanced covariates (Austin, 2011), meaning that the covariate distribution differs non-negligibly between high- and low-utilisation cases and that controlling for it is consequential for causal identification.

---

## 10. Shared Identification Assumptions

All causal estimates produced by this pipeline rest on three assumptions that cannot be verified from data alone and must be justified on subject-matter grounds:

1. **Backdoor criterion (no unobserved confounding).** The covariate set $X = \{\text{hour, dayofweek, n\_alternatives, res\_*}\}$ blocks all confounding paths between $T$ and $Y_k$. Any variable that causally influences both resource utilisation and routing choice but is absent from $X$ — such as case priority, queue length, or worker experience — would bias all estimates. The backdoor tests (Section 9) probe the plausibility of this assumption without being able to guarantee it.

2. **Stable Unit Treatment Value Assumption (SUTVA).** The routing decision of case $i$ does not causally depend on the treatment or outcome of any other case $j$, and there is only one version of each treatment level. SUTVA is threatened if, for example, the high utilisation of case $i$ directly displaces resources available to case $j$ in a way that is not captured by $j$'s own utilisation reading.

3. **Positivity (overlap).** Every combination of covariate values $X$ must have non-zero probability of occurring at any utilisation level $T = t$. If certain resources only ever operate at high utilisation (e.g. a SYSTEM resource that is always at 100%), the causal effect for those resources cannot be identified from the observational data, and extrapolation from the forest would be unreliable. The covariate balance SMDs partially diagnose this by flagging systematic divergence between high- and low-utilisation groups.

---

## References

- Athey, S., and Imbens, G. W. (2016). Recursive partitioning for heterogeneous causal effects. *Proceedings of the National Academy of Sciences*, 113(27), 7353–7360.
- Athey, S., Tibshirani, J., and Wager, S. (2019). Generalized random forests. *The Annals of Statistics*, 47(2), 1148–1178.
- Austin, P. C. (2011). An introduction to propensity score methods for reducing the effects of confounding in observational studies. *Multivariate Behavioral Research*, 46(3), 399–424.
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., and Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1–C68.
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.
- Robinson, P. M. (1988). Root-N-consistent semiparametric regression. *Econometrica*, 56(4), 931–954.
- Wager, S., and Athey, S. (2018). Estimation and inference of heterogeneous treatment effects using random forests. *Journal of the American Statistical Association*, 113(523), 1228–1242.
- Wager, S., Hastie, T., and Efron, B. (2014). Confidence intervals for random forests: the jackknife and the infinitesimal jackknife. *Journal of Machine Learning Research*, 15, 1625–1651.
- Zhao, Q., and Hastie, T. (2021). Causal interpretations of black-box models. *Journal of Business and Economic Statistics*, 39(1), 272–281.
