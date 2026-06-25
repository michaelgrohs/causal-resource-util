"""
causal_analysis.py
==================
Causal inference toolkit for Petri net routing decisions in .exs files.

The treatment is *continuous* resource utilisation in [0, 1].
The outcome is which transition fired, handled one-vs-rest per choice set.

Pipeline
--------
1. extract_decision_points   – filter to XOR decision rows, add covariates
2. logistic_regression_test  – conditional independence / association test
3. double_ml_test            – Debiased ML (Chernozhukov et al. 2018)
4. causal_forest_dml         – CausalForestDML (Wager & Athey 2018)
                               Returns ATE table + per-sample CATEs + tree
5. conditional_dose_response – CADR curves E[Y(t)−Y(0)|X] over a t-grid
6. backdoor_check            – Three identification-adequacy checks

Each function analyses every (choice_set, transition) pair independently.
Requires: econml, sklearn, scipy
"""

from __future__ import annotations

import warnings
warnings.filterwarnings('ignore')
import textwrap
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor, export_text

try:
    from econml.dml import CausalForestDML
    HAS_ECONML = True
except ImportError:
    HAS_ECONML = False


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

# Base features that are always present; resource one-hot columns are added
# dynamically by extract_decision_points and discovered via get_feature_cols().
DEFAULT_FEATURES_BASE = ["hour", "dayofweek", "n_alternatives"]


def _require_econml():
    if not HAS_ECONML:
        raise ImportError("This function requires econml.  pip install econml")


def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour and dayofweek columns."""
    out = df.copy()
    if "time_of_execution" in out.columns and out["time_of_execution"].notna().any():
        out["hour"]      = out["time_of_execution"].dt.hour.fillna(0).astype(int)
        out["dayofweek"] = out["time_of_execution"].dt.dayofweek.fillna(0).astype(int)
    else:
        out["hour"]      = 0
        out["dayofweek"] = 0
    return out


def _add_resource_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the 'resource' column into res_<name> columns (float)."""
    if "resource" not in df.columns:
        return df
    dummies = (
        pd.get_dummies(df["resource"].astype(str), prefix="res", dummy_na=False)
        .astype(float)
    )
    # drop columns already present to avoid duplicates on repeated calls
    new_cols = [c for c in dummies.columns if c not in df.columns]
    return pd.concat([df, dummies[new_cols]], axis=1)


def get_feature_cols(decisions: pd.DataFrame) -> list[str]:
    """
    Return the full feature-column list for a decisions DataFrame returned by
    extract_decision_points().  Includes base temporal features, one-hot
    resource columns (res_*), and alternative-utilisation summary columns
    (alt_util_*) when the dataset contains that information.
    """
    res_cols     = sorted(c for c in decisions.columns if c.startswith("res_"))
    alt_util_cols = sorted(c for c in decisions.columns if c.startswith("alt_util_"))
    return DEFAULT_FEATURES_BASE + res_cols + alt_util_cols


def _fmt_alt_utils(feat_ok: list[str]) -> str:
    """Return a comma-separated string of alt_util suffixes from feat_ok."""
    acts = [c[len("alt_util_"):] for c in feat_ok if c.startswith("alt_util_")]
    return ", ".join(acts) if acts else "—"


def _tid_label_map(grp: pd.DataFrame) -> dict:
    """Return {transition_id: chosen_label} from a decisions group."""
    if "chosen_label" in grp.columns:
        return dict(zip(grp["chosen"], grp["chosen_label"]))
    return {}


def _cs_labels(grp: pd.DataFrame) -> str:
    """Return the choice_set_labels string for this group (same for all rows)."""
    if "choice_set_labels" in grp.columns and not grp.empty:
        return str(grp["choice_set_labels"].iloc[0])
    return ""


def _alt_util_col(tid: int, cols) -> str | None:
    """Return the alt_util column name for a specific transition id.

    Column names always have the form alt_util_{label}_t{tid}, so matching the
    _t{tid} suffix is unambiguous even when multiple transitions share a label.
    Returns None when alt_util columns are absent (old-format files).
    """
    suffix = f"_t{tid}"
    for c in cols:
        if c.startswith("alt_util_") and c.endswith(suffix):
            return c
    return None


def _impute_col_means(X: np.ndarray, fill=None) -> np.ndarray:
    """Fill NaN values in-place; returns the array.

    fill=None  → per-column mean imputation (default)
    fill=0.0   → fill with 0 (treats null util as zero workload)

    Used when alt_util_* disambiguation creates separate columns per transition
    id (e.g. alt_util_A_CANCELLED_t3 / alt_util_A_CANCELLED_t7) and only one
    of them is non-NaN for a given row within a choice-set group.
    """
    if not np.isnan(X).any():
        return X
    if fill is None:
        col_means = np.nanmean(X, axis=0)
        rows, cols = np.where(np.isnan(X))
        X[rows, cols] = col_means[cols]
    else:
        X[np.isnan(X)] = fill
    return X


# Keep a backward-compatible alias (resource_enc no longer created)
DEFAULT_FEATURES = DEFAULT_FEATURES_BASE


# ─────────────────────────────────────────────────────────────────────────────
# 0. Data preparation
# ─────────────────────────────────────────────────────────────────────────────

def extract_decision_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter a parse_exs() DataFrame to genuine XOR decision rows and add
    covariates needed by all downstream functions.

    A row is a decision point when:
      • resource_utilisation is a known numeric value
      • other_enabled is non-empty (≥1 alternative was enabled)

    Returns columns:
        chosen           – fired transition id (int)
        chosen_label     – activity label of the fired transition
        choice_set       – tuple of int transition ids (fired + visible alternatives)
        choice_set_labels– tuple of activity labels for display
        util, case, n_alternatives, hour, dayofweek, res_<R1>, ...
        and, when alt_utils is present in df: alt_util_<label>_t<tid> columns

    Using transition IDs (not activity labels) as identifiers ensures that two
    Petri net transitions with the same label (e.g. t5 and t11 both named
    "A_ACTIVATED") are kept separate — they live in different nodes of the net
    and may have different enabling conditions.

    Use get_feature_cols(decisions) to retrieve the full covariate list,
    including all one-hot resource and alt-util columns for this dataset.
    """
    # Build transition_id → activity label mapping from visible (non-silent) rows
    trans_to_label: dict = (
        df[df["activity"].notna()]
        .drop_duplicates("transition")
        .set_index("transition")["activity"]
        .astype(str)
        .to_dict()
    )

    silent_tids: set = set(
        df[df["activity"].isna()]["transition"].unique().tolist()
    )

    def _label(tid) -> str:
        if tid in silent_tids:
            return "τ"
        return trans_to_label.get(tid, f"t{tid}")

    has_alt_utils = "alt_utils" in df.columns

    rows = []
    for _, r in df.iterrows():
        activity = r.get("activity")
        if activity is None:
            continue          # silent transitions are never decision-makers
        others = r.get("other_enabled", [])
        if not others:
            continue
        # Include all enabled alternatives — visible and silent alike.
        # Silent transitions carry no resource utilisation (their alt_util columns
        # will be NaN and dropped by feat_ok), but they represent real routing
        # options at the choice point and belong in the choice set.
        others_visible = [int(o) for o in others]
        if not others_visible:
            continue

        util_raw = r.get("resource_utilisation")
        # Keep model-move rows (also_in_log=False, util=None) — they are genuine
        # XOR gateway decisions; util becomes NaN so analysis functions can
        # drop or impute them when fitting the T nuisance model.
        util = float(util_raw) if (util_raw is not None and not (isinstance(util_raw, float) and np.isnan(util_raw))) else np.nan

        # Primary identifiers: transition IDs (integers).
        # Keeps t5 and t11 (both labelled "A_ACTIVATED") in separate groups.
        chosen       = int(r["transition"])
        choice_set   = tuple(sorted([chosen] + others_visible))
        # Always use activity_t{tid} format so the TID is visible in every output.
        chosen_label      = f"{_label(chosen)}_t{chosen}"
        choice_set_labels = tuple(f"{_label(t)}_t{t}" for t in sorted(choice_set))

        row_dict = {
            "chosen":              chosen,
            "chosen_label":        chosen_label,
            "choice_set":          choice_set,
            "choice_set_labels":   choice_set_labels,
            "util":                util,
            "case":                r.get("case", np.nan),
            "n_alternatives":      len(choice_set),
            "resource":            r.get("resource"),
            "time_of_execution":   r.get("time_of_execution"),
        }

        if has_alt_utils:
            alt_utils_raw = r.get("alt_utils") or {}
            # Keep all alternatives, including those with None util (silent or
            # unresourced transitions).  None becomes np.nan so the column exists
            # in the decisions frame; feat_ok drops all-NaN columns per group.
            _au = {k: (v if v is not None else np.nan)
                   for k, v in alt_utils_raw.items()}
            # defensive: any enabled alternative absent from alt_utils_raw
            # (e.g. never fires, so no entry in the .exs util list) gets NaN
            for _tid in others_visible:
                if _tid not in _au:
                    _au[_tid] = np.nan
            row_dict["_alt_utils"] = _au
            row_dict["_fired_tid"] = int(r["transition"])

        rows.append(row_dict)

    if not rows:
        return pd.DataFrame()

    decisions = pd.DataFrame(rows)
    decisions["trace_rank"] = (
        decisions["case"].rank(method="dense").astype(int) - 1
    )
    decisions = _add_temporal_features(decisions)
    decisions = _add_resource_dummies(decisions)

    # Expand per-alternative utilisation into individual alt_util_{t_id} columns,
    # one column per transition id that ever appears as an enabled alternative.
    #
    # Three cases for a given (row, tid):
    #   • tid was an enabled alternative here → alt_utils[tid] is set, not NaN
    #   • tid was the FIRED transition here   → not in alt_utils → NaN → fill with util
    #   • tid was not in this row's choice set → not in alt_utils → NaN → leave as NaN
    #
    # The fired-transition fill uses _fired_tid so we target only case 2,
    # not case 3. Analysis functions drop per-group all-NaN columns before
    # building X, so transitions outside a choice set carry no spurious signal.
    if has_alt_utils:
        all_tids = sorted({
            tid
            for d in decisions["_alt_utils"]
            if isinstance(d, dict)
            for tid in d.keys()
        })
        for tid in all_tids:
            col = f"alt_util_{tid}"
            decisions[col] = decisions["_alt_utils"].apply(
                lambda d, t=tid: d.get(t, np.nan) if isinstance(d, dict) else np.nan
            )
            # fill only where tid was the fired transition (not absent from choice set)
            fired_mask = decisions["_fired_tid"] == tid
            decisions.loc[fired_mask, col] = decisions.loc[fired_mask, "util"]
        decisions = decisions.drop(columns=["_alt_utils", "_fired_tid"])

        # Rename alt_util_{tid} → alt_util_{activity_label} for readability.
        # Unlabeled tids keep a t{tid} suffix.
        # When multiple tids share the same activity label (common in DFG/FLW Petri nets
        # where the same activity name appears on several transitions), append _t{tid}
        # to disambiguate — otherwise pandas creates duplicate column names, which causes
        # grp[col] to return a DataFrame instead of a Series in downstream groupby loops.
        target_labels = {tid: trans_to_label.get(tid, f"t{tid}") for tid in all_tids}
        # Always include the tid suffix so downstream code can locate each
        # transition's column unambiguously via its _t{tid} ending.
        rename_map = {
            f"alt_util_{tid}": f"alt_util_{label}_t{tid}"
            for tid, label in target_labels.items()
        }
        decisions = decisions.rename(columns=rename_map)

    return decisions


# ─────────────────────────────────────────────────────────────────────────────
# 1. Logistic regression + permutation test
# ─────────────────────────────────────────────────────────────────────────────

def logistic_regression_test(
    decisions: pd.DataFrame,
    n_permutations: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Conditional independence test via logistic regression + label permutation.

    H₀: choice ⊥ util | choice_set
    """
    rng     = np.random.default_rng(seed)
    records = []

    for cs, grp in decisions.groupby("choice_set"):
        grp     = grp[grp["util"].notna()]
        n       = len(grp)
        X       = grp["util"].values
        y_raw   = grp["chosen"].values
        uniq    = np.unique(y_raw)
        t2l     = _tid_label_map(grp)
        base    = {"choice_set": str(cs), "choice_set_labels": _cs_labels(grp),
                   "n": n, "choices_seen": list(uniq)}

        if len(uniq) < 2:
            records.append({**base, "coef": np.nan, "ci_95": "—",
                             "perm_p_value": np.nan, "reject_H0": "—",
                             "note": f"only t{uniq[0]} ({t2l.get(uniq[0], '?')}) ever chosen"})
            continue

        le = LabelEncoder()
        y  = le.fit_transform(y_raw)

        def _fit(Xf, yf):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = LogisticRegression(max_iter=1_000, solver="lbfgs")
                m.fit(Xf.reshape(-1, 1), yf)
                return float(m.coef_[0, 0])

        obs   = _fit(X, y)
        boots = [_fit(X[idx], y[idx])
                 for _ in range(n_permutations)
                 if len(np.unique(y[idx := rng.integers(0, n, n)])) > 1]
        boots = np.array(boots)
        ci    = np.percentile(boots, [2.5, 97.5]) if len(boots) >= 20 else [np.nan, np.nan]

        perms = [_fit(X, rng.permutation(y)) for _ in range(n_permutations)
                 if len(np.unique(rng.permutation(y))) > 1]
        perms = np.array(perms)
        pval  = float(np.mean(np.abs(perms) >= abs(obs))) if len(perms) else np.nan

        hi_tid = le.classes_[-1]; lo_tid = le.classes_[0]
        hi_lbl = t2l.get(hi_tid, str(hi_tid)); lo_lbl = t2l.get(lo_tid, str(lo_tid))
        records.append({**base,
            "coef":         round(obs, 4),
            "ci_95":        f"[{ci[0]:.3f}, {ci[1]:.3f}]" if not np.isnan(ci[0]) else "—",
            "perm_p_value": round(pval, 4),
            "reject_H0":    bool(pval < 0.05) if not np.isnan(pval) else "—",
            "note": (f"higher util → t{hi_tid} ({hi_lbl}) over t{lo_tid} ({lo_lbl})"
                     if obs > 0 else
                     f"higher util → t{lo_tid} ({lo_lbl}) over t{hi_tid} ({hi_lbl})"),
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Double ML (Partially Linear Regression)
# ─────────────────────────────────────────────────────────────────────────────

def double_ml_test(
    decisions: pd.DataFrame,
    feature_cols: list[str] = DEFAULT_FEATURES,
    n_splits: int = 5,
    seed: int = 42,
    relative_t: bool = False,
    narrow_t_nuisance: bool = False,
    mean_t: bool = False,
    differential_t: bool = False,
    null_util_fill: str = "mean",
) -> pd.DataFrame:
    """
    Debiased / Double ML – Partially Linear Regression.

    Y_k − E[Y_k|X]  =  θ_k · (util − E[util|X])  +  ε

    Cross-fitted nuisances (Ridge) remove regularisation bias.
    HC3-robust SE.  Effect θ_k is on the probability scale.
    """
    rng_state = np.random.RandomState(seed)
    records   = []

    for cs, grp in decisions.groupby("choice_set"):
        grp     = grp[grp["util"].notna()]
        n       = len(grp)
        choices = grp["chosen"].values
        uniq    = np.unique(choices)
        t2l     = _tid_label_map(grp)
        base    = {"choice_set": str(cs), "choice_set_labels": _cs_labels(grp),
                   "n": n, "choices_seen": list(uniq)}

        if len(uniq) < 2 or n < 2 * n_splits:
            records.append({**base, "transition": "—", "transition_label": "—",
                             "alt_utils": "—",
                             "theta": np.nan, "se": np.nan, "ci_95": "—",
                             "p_value": np.nan, "reject_H0": "—",
                             "note": "too few observations"})
            continue

        kf = KFold(n_splits=min(n_splits, n // 2), shuffle=True,
                   random_state=rng_state.randint(0, 10_000))

        for trans in uniq:
            # T = trans's own utilisation (alt_util_*_t{tid}), always observed
            #     regardless of which transition fired. Falls back to grp["util"]
            #     for old-format files that have no alt_util columns.
            # X = all other alt_util columns + temporal/resource features.
            # relative_t=True: T becomes T_own − mean(other alt_utils), removing
            #     shared system-load variation and fixing the collinearity with X.
            T_col   = _alt_util_col(trans, feature_cols)
            T       = grp[T_col].values.astype(float) if T_col else grp["util"].values.astype(float)
            feat_ok = [c for c in feature_cols
                       if c in grp.columns and c != T_col and grp[c].notna().any()]
            if relative_t and T_col:
                _other_alts = [c for c in feat_ok if c.startswith("alt_util_")]
                if _other_alts:
                    T = T - np.nanmean(grp[_other_alts].values.astype(float), axis=1)
            if mean_t and T_col:
                _cs_alts = [T_col] + [c for c in feat_ok if c.startswith("alt_util_")]
                T = np.nanmean(grp[_cs_alts].values.astype(float), axis=1)
                feat_ok = [c for c in feat_ok if not c.startswith("alt_util_")]
            _mean_other = None
            if differential_t and T_col:
                _other_alts = [c for c in feat_ok if c.startswith("alt_util_")]
                if _other_alts:
                    _mean_other = np.nanmean(grp[_other_alts].values.astype(float), axis=1)
                    _mean_other = np.where(np.isnan(_mean_other),
                                           np.nanmean(_mean_other), _mean_other)
                    T = T - _mean_other
                    feat_ok = [c for c in feat_ok if not c.startswith("alt_util_")]

            _fill = None if null_util_fill == "mean" else 0.0
            if _mean_other is not None:
                _base_X = _impute_col_means(grp[feat_ok].values.astype(float), _fill)
                X_cov = np.column_stack([_mean_other.reshape(-1, 1), _base_X])
            else:
                X_cov = _impute_col_means(grp[feat_ok].values.astype(float), _fill)
            alt_used = _fmt_alt_utils(feat_ok)

            if narrow_t_nuisance and not differential_t:
                _feat_T = [c for c in feat_ok if not c.startswith("alt_util_")]
                X_T_cov = _impute_col_means(grp[_feat_T].values.astype(float), _fill) if _feat_T else X_cov
            else:
                X_T_cov = X_cov

            tt_res = np.zeros(n)
            for tr, te in kf.split(X_T_cov):
                m = Ridge(alpha=1.0).fit(X_T_cov[tr], T[tr])
                tt_res[te] = T[te] - m.predict(X_T_cov[te])

            denom = float(tt_res @ tt_res)
            if denom < 1e-12:
                continue

            y     = (choices == trans).astype(float)
            y_res = np.zeros(n)
            for tr, te in kf.split(X_cov):
                m = Ridge(alpha=1.0).fit(X_cov[tr], y[tr])
                y_res[te] = y[te] - m.predict(X_cov[te])

            theta    = float((tt_res @ y_res) / denom)
            resid    = y_res - theta * tt_res
            lev      = tt_res ** 2 / denom
            hc3_w    = resid ** 2 / np.clip(1 - lev, 1e-6, None) ** 2
            var_th   = float(np.sum(hc3_w * tt_res ** 2) / denom ** 2)
            se       = float(np.sqrt(max(var_th, 0)))
            ci_lo, ci_hi = theta - 1.96 * se, theta + 1.96 * se
            z        = theta / se if se > 1e-12 else np.nan
            pval     = float(2 * stats.norm.sf(abs(z))) if not np.isnan(z) else np.nan

            trans_lbl = t2l.get(trans, str(trans))
            records.append({**base,
                "transition":       trans,
                "transition_label": trans_lbl,
                "alt_utils":        alt_used,
                "theta":      round(theta, 6),
                "se":         round(se, 6),
                "ci_95":      f"[{ci_lo:.4f}, {ci_hi:.4f}]",
                "p_value":    round(pval, 4),
                "reject_H0":  bool(pval < 0.05) if not np.isnan(pval) else "—",
                "note": (f"θ={theta:+.4f}: 1-unit ↑ util "
                         f"{'↑' if theta > 0 else '↓'} P(t{trans}/{trans_lbl}) by {abs(theta):.1%}"),
            })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Artifact container returned by causal_forest_dml
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _ForestArtifact:
    choice_set:   str
    transition:   str
    n:            int
    feature_cols: list[str]
    X:            np.ndarray
    T:            np.ndarray
    Y:            np.ndarray
    model:        object          # fitted CausalForestDML
    cate:         np.ndarray      # per-sample CATE
    cate_lb:      np.ndarray
    cate_ub:      np.ndarray
    tree:         DecisionTreeRegressor
    tree_text:    str
    feat_imp:     pd.Series


# ─────────────────────────────────────────────────────────────────────────────
# 3. Causal Forest DML
# ─────────────────────────────────────────────────────────────────────────────

class _ColumnSubsetRegressor:
    """Sklearn-compatible wrapper that fits on a column subset of X.

    Used by causal_forest_dml with narrow_t_nuisance=True so that model_t
    sees only non-alt_util columns while model_y uses the full X.
    """
    def __init__(self, base, col_indices):
        self.base = base
        self.cols = col_indices

    def fit(self, X, y, sample_weight=None):
        kw = {"sample_weight": sample_weight} if sample_weight is not None else {}
        self.base.fit(X[:, self.cols], y, **kw)
        return self

    def predict(self, X):
        return self.base.predict(X[:, self.cols])


def causal_forest_dml(
    decisions: pd.DataFrame,
    feature_cols: list[str] = DEFAULT_FEATURES,
    n_estimators: int = 500,
    min_samples_leaf: int = 10,
    tree_depth: int = 3,
    seed: int = 42,
    verbose: bool = True,
    relative_t: bool = False,
    narrow_t_nuisance: bool = False,
    mean_t: bool = False,
    differential_t: bool = False,
    null_util_fill: str = "mean",
) -> tuple[pd.DataFrame, list[_ForestArtifact]]:
    """
    CausalForestDML with Debiased ML nuisances (Wager & Athey 2018 /
    Chernozhukov et al. 2018).

    For each (choice_set, transition) pair in one-vs-rest fashion:

      Y_k = 1(chose transition k),  T = util (continuous),  X = feature matrix

    The forest uses honest splitting (separate samples for splits and leaf
    estimates) which yields valid confidence intervals without permutation.

    Nuisance models: GradientBoosting for both E[Y|X] and E[T|X].

    After fitting, an interpretable DecisionTreeRegressor is grown on the
    per-sample CATE estimates with X as features, and its text representation
    is printed.

    Parameters
    ----------
    decisions     : output of extract_decision_points()
    feature_cols  : covariate columns to use as X
    n_estimators  : number of trees in the causal forest
    min_samples_leaf : minimum leaf size (larger → smoother, less overfit)
    tree_depth    : max depth of the interpretable summary tree
    verbose       : print summary trees

    Returns
    -------
    summary_df  : DataFrame with ATE, CI, p-value per (choice_set, transition)
    artifacts   : list of _ForestArtifact — fitted models + arrays needed by
                  conditional_dose_response() and backdoor_check()
    """
    _require_econml()

    summary_rows = []
    artifacts    = []

    for cs, grp in decisions.groupby("choice_set"):
        grp     = grp[grp["util"].notna()]
        n       = len(grp)
        choices = grp["chosen"].values
        uniq    = np.unique(choices)
        cs_str  = str(cs)
        t2l     = _tid_label_map(grp)
        base    = {"choice_set": cs_str, "choice_set_labels": _cs_labels(grp),
                   "n": n, "choices_seen": list(uniq)}

        if len(uniq) < 2 or n < 30:
            summary_rows.append({**base, "transition": "—", "transition_label": "—",
                                  "alt_utils": "—",
                                  "ate": np.nan, "ci_95": "—", "p_value": np.nan,
                                  "reject_H0": "—", "note": "too few observations"})
            continue

        for trans in uniq:
            trans_lbl = t2l.get(trans, str(trans))

            # T = trans's own utilisation (alt_util_*_t{tid}), always observed
            #     regardless of which transition fired. Falls back to grp["util"]
            #     for old-format files that have no alt_util columns.
            # X = all other alt_util columns + temporal/resource features.
            # relative_t=True: T becomes T_own − mean(other alt_utils), removing
            #     shared system-load variation and fixing the collinearity with X.
            T_col   = _alt_util_col(trans, feature_cols)
            T       = grp[T_col].values.astype(float) if T_col else grp["util"].values.astype(float)
            feat_ok = [c for c in feature_cols
                       if c in grp.columns and c != T_col and grp[c].notna().any()]
            if relative_t and T_col:
                _other_alts = [c for c in feat_ok if c.startswith("alt_util_")]
                if _other_alts:
                    T = T - np.nanmean(grp[_other_alts].values.astype(float), axis=1)
            if mean_t and T_col:
                _cs_alts = [T_col] + [c for c in feat_ok if c.startswith("alt_util_")]
                T = np.nanmean(grp[_cs_alts].values.astype(float), axis=1)
                feat_ok = [c for c in feat_ok if not c.startswith("alt_util_")]
            _mean_other = None
            if differential_t and T_col:
                _other_alts = [c for c in feat_ok if c.startswith("alt_util_")]
                if _other_alts:
                    _mean_other = np.nanmean(grp[_other_alts].values.astype(float), axis=1)
                    _mean_other = np.where(np.isnan(_mean_other),
                                           np.nanmean(_mean_other), _mean_other)
                    T = T - _mean_other
                    feat_ok = ["mean_other_utils"] + [c for c in feat_ok
                                                      if not c.startswith("alt_util_")]

            _fill = None if null_util_fill == "mean" else 0.0
            if _mean_other is not None:
                _base_X = _impute_col_means(grp[feat_ok[1:]].values.astype(float), _fill)
                X = np.column_stack([_mean_other.reshape(-1, 1), _base_X])
            else:
                X = _impute_col_means(grp[feat_ok].values.astype(float), _fill)

            # Drop constant columns — zero variance makes the moment matrix singular
            col_var   = X.var(axis=0)
            live_mask = col_var > 1e-10
            if not live_mask.all():
                feat_ok = [f for f, ok in zip(feat_ok, live_mask) if ok]
                X       = X[:, live_mask]

            alt_used = _fmt_alt_utils(feat_ok)

            if len(feat_ok) == 0 or T.var() < 1e-10:
                summary_rows.append({**base, "transition": trans,
                                      "transition_label": trans_lbl,
                                      "alt_utils": alt_used,
                                      "ate": np.nan, "ci_95": "—", "p_value": np.nan,
                                      "reject_H0": "—",
                                      "note": "no variance in features or treatment"})
                continue

            Y = (choices == trans).astype(float)

            if narrow_t_nuisance and not differential_t:
                _narrow_idx = [i for i, c in enumerate(feat_ok)
                               if not c.startswith("alt_util_")]
                _model_t = _ColumnSubsetRegressor(
                    GradientBoostingRegressor(n_estimators=100, random_state=seed),
                    _narrow_idx if _narrow_idx else list(range(len(feat_ok))),
                )
            else:
                _model_t = GradientBoostingRegressor(n_estimators=100, random_state=seed)

            cf = CausalForestDML(
                model_y=GradientBoostingRegressor(n_estimators=100, random_state=seed),
                model_t=_model_t,
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
                discrete_treatment=False,
                random_state=seed,
            )
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cf.fit(Y, T, X=X)
            except np.linalg.LinAlgError as e:
                summary_rows.append({**base, "transition": trans,
                                      "transition_label": trans_lbl,
                                      "alt_utils": alt_used,
                                      "ate": np.nan, "ci_95": "—", "p_value": np.nan,
                                      "reject_H0": "—",
                                      "note": f"singular matrix — skipped ({e})"})
                continue

            # Per-sample CATE
            cate = cf.effect(X)
            lb, ub = cf.effect_interval(X, alpha=0.05)

            # ATE with CI
            inf = cf.ate_inference(X=X)
            ate = float(inf.mean_point)
            ci  = inf.conf_int_mean()
            # derive SE and p-value from the 95 % CI (avoids version-specific attrs)
            se_ate = (float(ci[1]) - float(ci[0])) / (2 * 1.96)
            z_ate  = ate / se_ate if se_ate > 1e-12 else 0.0
            pval   = float(2 * stats.norm.sf(abs(z_ate)))

            # ── Interpretable summary tree on CATEs ───────────────────────────
            tree = DecisionTreeRegressor(max_depth=tree_depth, random_state=seed)
            tree.fit(X, cate)
            feat_imp = pd.Series(tree.feature_importances_, index=feat_ok,
                                 name=f"importance_{trans}")
            tree_txt = (
                f"\n{'='*60}\n"
                f"CATE tree  |  choice_set={cs_str}  |  t{trans} ({trans_lbl})\n"
                f"(leaf value = estimated CATE; +ve → higher util raises P(t{trans}/{trans_lbl}))\n"
                f"{'='*60}\n"
                + export_text(tree, feature_names=feat_ok)
            )

            if verbose:
                print(tree_txt)
                print(f"Feature importances:\n{feat_imp.sort_values(ascending=False)}\n")

            artifacts.append(_ForestArtifact(
                choice_set=cs_str, transition=trans, n=n,
                feature_cols=feat_ok, X=X, T=T, Y=Y,
                model=cf, cate=cate, cate_lb=lb, cate_ub=ub,
                tree=tree, tree_text=tree_txt, feat_imp=feat_imp,
            ))

            summary_rows.append({**base,
                "transition":       trans,
                "transition_label": trans_lbl,
                "alt_utils":        alt_used,
                "ate":        round(ate, 6),
                "ci_95":      f"[{float(ci[0]):.4f}, {float(ci[1]):.4f}]",
                "p_value":    round(pval, 4),
                "reject_H0":  bool(pval < 0.05),
                "note": (f"ATE={ate:+.4f}: 0→1 util "
                         f"{'↑' if ate > 0 else '↓'} P(t{trans}/{trans_lbl}) by {abs(ate):.1%}"),
            })

    return pd.DataFrame(summary_rows), artifacts


# ─────────────────────────────────────────────────────────────────────────────
# 4. Conditional Average Dose Response (CADR)
# ─────────────────────────────────────────────────────────────────────────────

def conditional_dose_response(
    artifacts: list[_ForestArtifact],
    t_grid: Optional[np.ndarray] = None,
    t_ref: float = 0.0,
    subgroup_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Estimate Conditional Average Dose-Response curves from fitted CausalForestDML
    models.  Implements the marginal-effect approach:

        CADR(t) = E_X[ effect(T0=t_ref, T1=t) | X ]

    For each t in t_grid, we evaluate the average causal effect of setting
    utilisation to t relative to the baseline t_ref.  Confidence intervals
    come from effect_interval().

    Optionally stratified by a subgroup column in the original decisions frame
    (e.g. "resource_enc" to get one curve per resource type).

    Parameters
    ----------
    artifacts     : list returned by causal_forest_dml()
    t_grid        : treatment levels to evaluate (default: 20 points in [0,1])
    t_ref         : baseline reference treatment level
    subgroup_col  : index into artifact.feature_cols to stratify by

    Returns
    -------
    DataFrame with columns: choice_set, transition, t, mean_effect,
                             ci_lb, ci_ub, [subgroup]
    """
    _require_econml()

    if t_grid is None:
        t_grid = np.linspace(0.0, 1.0, 20)

    rows = []

    for art in artifacts:
        cf     = art.model
        X      = art.X
        n      = len(X)

        # Optionally split X by subgroup
        if subgroup_col and subgroup_col in art.feature_cols:
            col_idx    = art.feature_cols.index(subgroup_col)
            subgroups  = {str(v): X[:, col_idx] == v
                          for v in np.unique(X[:, col_idx])}
        else:
            subgroups = {"all": np.ones(n, dtype=bool)}

        for sg_name, mask in subgroups.items():
            X_sub = X[mask]
            if len(X_sub) < 5:
                continue

            for t in t_grid:
                T0 = np.full(len(X_sub), t_ref)
                T1 = np.full(len(X_sub), t)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    eff    = cf.effect(X_sub, T0=T0, T1=T1)
                    lb, ub = cf.effect_interval(X_sub, T0=T0, T1=T1, alpha=0.05)

                rows.append({
                    "choice_set":  art.choice_set,
                    "transition":  art.transition,
                    "subgroup":    sg_name,
                    "t":           round(t, 4),
                    "mean_effect": round(float(eff.mean()), 6),
                    "ci_lb":       round(float(lb.mean()), 6),
                    "ci_ub":       round(float(ub.mean()), 6),
                })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Backdoor adequacy checks
# ─────────────────────────────────────────────────────────────────────────────

def backdoor_check(
    decisions: pd.DataFrame,
    feature_cols: list[str] = DEFAULT_FEATURES,
    n_splits: int = 5,
    n_placebo: int = 200,
    seed: int = 42,
    relative_t: bool = False,
    narrow_t_nuisance: bool = False,
    mean_t: bool = False,
    differential_t: bool = False,
    null_util_fill: str = "mean",
) -> pd.DataFrame:
    """
    Three complementary checks for whether the backdoor criterion is plausibly
    satisfied — i.e., whether X is sufficient to block all confounding paths
    between T (util) and Y (choice).

    Check 1 – Nuisance R² (propensity of treatment)
    ─────────────────────────────────────────────────
    Cross-validated R² of predicting util from X.
    • High R²: X strongly predicts util → good control for confounders that
      also predict util.
    • R² near 0: util is nearly random given X → weak overlap concern but
      also little confounding via X.
    Interpretation: neither extreme is bad per se; what matters is that the
    same X that predicts util also captures the confounders.

    Check 2 – Placebo treatment test
    ──────────────────────────────────
    Replace T with a random permutation of util, re-estimate θ with Double ML.
    Under the null (no causal effect survives after controlling for X), the
    placebo θ should be near zero.  A large placebo θ suggests spurious
    correlation not absorbed by X.

    Check 3 – Covariate balance (standardised mean difference)
    ─────────────────────────────────────────────────────────────
    Compare X between "high util" (≥ median) and "low util" (< median) groups.
    Large SMD (> 0.1 is a conventional threshold) indicates that treatment
    assignment is associated with covariates in ways that could introduce bias.

    Returns
    -------
    DataFrame with one row per (choice_set, check_name, feature/statistic).
    """
    rng       = np.random.default_rng(seed)
    rng_state = np.random.RandomState(seed)
    records   = []

    for cs, grp in decisions.groupby("choice_set"):
        grp     = grp[grp["util"].notna()]
        n       = len(grp)
        choices = grp["chosen"].values
        uniq    = np.unique(choices)
        cs_str  = str(cs)
        cs_lbls = _cs_labels(grp)
        t2l     = _tid_label_map(grp)

        if n < 20:
            records.append({"choice_set": cs_str, "choice_set_labels": cs_lbls,
                            "transition": "—", "transition_label": "—",
                            "alt_utils": "—", "check": "skipped",
                            "detail": "—", "value": np.nan,
                            "flag": "—", "note": "too few observations"})
            continue

        for trans in uniq:
            trans_lbl = t2l.get(trans, str(trans))

            # T = trans's own utilisation (alt_util_*_t{tid}), always observed
            #     regardless of which transition fired. Falls back to grp["util"]
            #     for old-format files that have no alt_util columns.
            # X = all other alt_util columns + temporal/resource features.
            # relative_t=True: T becomes T_own − mean(other alt_utils), removing
            #     shared system-load variation and fixing the collinearity with X.
            T_col   = _alt_util_col(trans, feature_cols)
            T       = grp[T_col].values.astype(float) if T_col else grp["util"].values.astype(float)
            feat_ok = [c for c in feature_cols
                       if c in grp.columns and c != T_col and grp[c].notna().any()]
            if relative_t and T_col:
                _other_alts = [c for c in feat_ok if c.startswith("alt_util_")]
                if _other_alts:
                    T = T - np.nanmean(grp[_other_alts].values.astype(float), axis=1)
            if mean_t and T_col:
                _cs_alts = [T_col] + [c for c in feat_ok if c.startswith("alt_util_")]
                T = np.nanmean(grp[_cs_alts].values.astype(float), axis=1)
                feat_ok = [c for c in feat_ok if not c.startswith("alt_util_")]
            _mean_other = None
            if differential_t and T_col:
                _other_alts = [c for c in feat_ok if c.startswith("alt_util_")]
                if _other_alts:
                    _mean_other = np.nanmean(grp[_other_alts].values.astype(float), axis=1)
                    _mean_other = np.where(np.isnan(_mean_other),
                                           np.nanmean(_mean_other), _mean_other)
                    T = T - _mean_other
                    feat_ok = ["mean_other_utils"] + [c for c in feat_ok
                                                      if not c.startswith("alt_util_")]

            if len(feat_ok) == 0:
                records.append({"choice_set": cs_str, "choice_set_labels": cs_lbls,
                                "transition": trans, "transition_label": trans_lbl,
                                "alt_utils": "—", "check": "skipped",
                                "detail": "—", "value": np.nan,
                                "flag": "—", "note": "no features available"})
                continue

            _fill = None if null_util_fill == "mean" else 0.0
            if _mean_other is not None:
                _base_X = _impute_col_means(grp[feat_ok[1:]].values.astype(float), _fill)
                X = np.column_stack([_mean_other.reshape(-1, 1), _base_X])
            else:
                X = _impute_col_means(grp[feat_ok].values.astype(float), _fill)
            alt_used = _fmt_alt_utils(feat_ok)
            base_row = {"choice_set": cs_str, "choice_set_labels": cs_lbls,
                        "transition": trans, "transition_label": trans_lbl,
                        "alt_utils": alt_used}

            if narrow_t_nuisance and not differential_t:
                _feat_T = [c for c in feat_ok if not c.startswith("alt_util_")]
                X_T = _impute_col_means(grp[_feat_T].values.astype(float), _fill) if _feat_T else X
            else:
                X_T = X

            # ── Check 1: Nuisance R² ─────────────────────────────────────────
            kf    = KFold(n_splits=min(n_splits, n // 2), shuffle=True,
                          random_state=rng_state.randint(0, 10_000))
            T_hat = cross_val_predict(Ridge(alpha=1.0), X_T, T, cv=kf)
            ss_res = np.sum((T - T_hat) ** 2)
            ss_tot = np.sum((T - T.mean()) ** 2)
            r2     = 1 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan

            records.append({**base_row, "check": "nuisance_R2",
                "detail": "R²(util_k ~ X, cross-val)",
                "value": round(float(r2), 4),
                "flag": ("✓ X predicts util well" if r2 > 0.1
                         else "△ X barely predicts util — confounding via X is minimal"),
                "note": ("High R² is good: X controls util variation.\n"
                         "Low R² suggests util is (near-)random given X."),
            })

            # ── Check 2: Placebo treatment ───────────────────────────────────
            kf2 = KFold(n_splits=min(n_splits, n // 2), shuffle=True,
                        random_state=rng_state.randint(0, 10_000))
            placebo_thetas = []
            y_bin = (choices == trans).astype(float)
            for _ in range(n_placebo):
                T_perm  = rng.permutation(T)
                t_res_p = np.zeros(n)
                for tr, te in kf2.split(X_T):
                    m = Ridge(alpha=1.0).fit(X_T[tr], T_perm[tr])
                    t_res_p[te] = T_perm[te] - m.predict(X_T[te])
                denom_p = float(t_res_p @ t_res_p)
                if denom_p < 1e-12:
                    continue
                y_res_p = np.zeros(n)
                for tr, te in kf2.split(X):
                    m = Ridge(alpha=1.0).fit(X[tr], y_bin[tr])
                    y_res_p[te] = y_bin[te] - m.predict(X[te])
                placebo_thetas.append(float((t_res_p @ y_res_p) / denom_p))

            placebo_thetas = np.array(placebo_thetas)
            p_mean = float(placebo_thetas.mean()) if len(placebo_thetas) else np.nan
            p_sd   = float(placebo_thetas.std())  if len(placebo_thetas) else np.nan

            records.append({**base_row, "check": "placebo_treatment",
                "detail": f"mean|std of θ under {n_placebo} random-T permutations",
                "value": round(p_mean, 6),
                "flag": ("✓ placebo θ ≈ 0 — no spurious signal"
                         if abs(p_mean) < 2 * p_sd
                         else "✗ large placebo θ — potential uncontrolled confounding"),
                "note": (f"mean={p_mean:.4f}, sd={p_sd:.4f}.  "
                         "Should be near zero if X absorbs confounders."),
            })

            # ── Check 3: Covariate balance (SMD) ────────────────────────────
            t_lo = float(T.min())
            t_hi = float(T.max())

            if t_hi - t_lo < 1e-10:
                # T is truly constant — no variation to split on
                for col in feat_ok:
                    records.append({**base_row,
                        "check": "covariate_balance_SMD", "detail": col,
                        "value": np.nan,
                        "flag": "— skipped (T constant in this group)",
                        "note": "SMD requires T variation; all observations share the same utilisation.",
                    })
            else:
                # Split on median.  When the median equals the floor (>50 % of values
                # are at the minimum, e.g. many T=0 rows), a >= split is degenerate:
                # every row goes into "hi".  Fall back to a strict-inequality split on
                # the minimum value so the floor observations form the "low" group.
                split_val = float(np.median(T))
                if split_val - t_lo < 1e-10:
                    hi_mask = T > t_lo
                else:
                    hi_mask = T >= split_val
                hi = X[ hi_mask]
                lo = X[~hi_mask]

                for j, col in enumerate(feat_ok):
                    hi_j, lo_j = hi[:, j], lo[:, j]
                    pooled_sd  = np.sqrt((hi_j.var() + lo_j.var()) / 2 + 1e-12)
                    smd        = float((hi_j.mean() - lo_j.mean()) / pooled_sd)
                    records.append({**base_row,
                        "check": "covariate_balance_SMD", "detail": col,
                        "value": round(smd, 4),
                        "flag": ("✓ balanced" if abs(smd) < 0.1
                                 else "△ imbalanced (|SMD|>0.1)"),
                        "note": ("SMD = (mean_hi − mean_lo) / pooled_sd. "
                                 "|SMD| < 0.1 is the conventional balance threshold."),
                })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Multivariate-treatment Double ML
# ─────────────────────────────────────────────────────────────────────────────

def multi_dml_test(
    decisions: pd.DataFrame,
    feature_cols: list[str] = DEFAULT_FEATURES,
    t_spec: str = "mean",
    n_splits: int = 5,
    seed: int = 42,
    null_util_fill: str = "mean",
) -> pd.DataFrame:
    """
    Double ML with a multivariate treatment vector T.

    X is always restricted to temporal + resource features (alt_util cols are
    moved into T, so they must not appear in X too).

    t_spec="mean"
        T = [u_k,  mean(u_{j≠k})]
        Two components: own util and system-wide load level.
        θ_1 answers "does u_k matter beyond load level?";
        θ_2 answers "does the load level itself shift the choice probability?".

    t_spec="all"
        T = [u_k,  u_{j1},  u_{j2}, …]
        Each competitor util gets its own coefficient (cross-effects).
        Useful to test whether individual competitor load matters, but the
        T_res matrix can be ill-conditioned when utils move together.

    HC3-robust SE via the multivariate sandwich estimator.
    Returns one row per (choice_set, transition, t_component).
    """
    if t_spec not in ("mean", "all"):
        raise ValueError("t_spec must be 'mean' or 'all'")

    rng_state = np.random.RandomState(seed)
    records   = []

    for cs, grp in decisions.groupby("choice_set"):
        grp     = grp[grp["util"].notna()]
        n       = len(grp)
        choices = grp["chosen"].values
        uniq    = np.unique(choices)
        t2l     = _tid_label_map(grp)
        base    = {"choice_set": str(cs), "choice_set_labels": _cs_labels(grp),
                   "n": n, "choices_seen": list(uniq)}

        if len(uniq) < 2 or n < 2 * n_splits:
            records.append({**base, "transition": "—", "transition_label": "—",
                             "t_component": "—", "alt_utils": "—",
                             "theta": np.nan, "se": np.nan, "ci_95": "—",
                             "p_value": np.nan, "reject_H0": "—",
                             "note": "too few observations"})
            continue

        kf = KFold(n_splits=min(n_splits, n // 2), shuffle=True,
                   random_state=rng_state.randint(0, 10_000))

        for trans in uniq:
            trans_lbl = t2l.get(trans, str(trans))
            T_col     = _alt_util_col(trans, feature_cols)

            if not T_col:
                continue   # old-format file without alt_util cols

            # X: temporal + resource only — no alt_util cols
            feat_ok = [c for c in feature_cols
                       if c in grp.columns
                       and c != T_col
                       and not c.startswith("alt_util_")
                       and grp[c].notna().any()]

            other_alts = [c for c in feature_cols
                          if c in grp.columns
                          and c.startswith("alt_util_")
                          and c != T_col
                          and grp[c].notna().any()]

            if not other_alts or len(feat_ok) == 0:
                continue

            T_own = grp[T_col].values.astype(float)

            if t_spec == "mean":
                mean_other = np.nanmean(grp[other_alts].values.astype(float), axis=1)
                mean_other = np.where(np.isnan(mean_other),
                                      np.nanmean(mean_other), mean_other)
                T_vec    = np.column_stack([T_own, mean_other])
                t_labels = ["u_k", "mean_other"]
            else:  # "all"
                _fill = None if null_util_fill == "mean" else 0.0
                T_others = _impute_col_means(grp[other_alts].values.astype(float), _fill)
                T_vec    = np.column_stack([T_own, T_others])
                t_labels = ["u_k"] + other_alts

            _fill = None if null_util_fill == "mean" else 0.0
            X_cov    = _impute_col_means(grp[feat_ok].values.astype(float), _fill)
            alt_used = ", ".join(t_labels)

            # Cross-fit residuals for every T component
            p     = T_vec.shape[1]
            T_res = np.zeros_like(T_vec)
            for j in range(p):
                if T_vec[:, j].std() < 1e-10:
                    continue   # constant component — leave residual at zero
                for tr, te in kf.split(X_cov):
                    m = Ridge(alpha=1.0).fit(X_cov[tr], T_vec[tr, j])
                    T_res[te, j] = T_vec[te, j] - m.predict(X_cov[te])

            M = T_res.T @ T_res
            if np.linalg.matrix_rank(M) < p or np.linalg.cond(M) > 1e10:
                records.append({**base,
                    "transition": trans, "transition_label": trans_lbl,
                    "t_component": "—", "alt_utils": alt_used,
                    "theta": np.nan, "se": np.nan, "ci_95": "—",
                    "p_value": np.nan, "reject_H0": "—",
                    "note": f"T_res matrix ill-conditioned (cond={np.linalg.cond(M):.1e})"})
                continue

            M_inv = np.linalg.inv(M)

            # Y residuals
            y     = (choices == trans).astype(float)
            y_res = np.zeros(n)
            for tr, te in kf.split(X_cov):
                m = Ridge(alpha=1.0).fit(X_cov[tr], y[tr])
                y_res[te] = y[te] - m.predict(X_cov[te])

            theta_vec = M_inv @ (T_res.T @ y_res)

            # HC3 sandwich SE
            resid = y_res - T_res @ theta_vec
            h     = np.einsum("ij,jk,ik->i", T_res, M_inv, T_res)  # per-sample leverage
            hc3_w = resid ** 2 / np.clip(1 - h, 1e-6, None) ** 2
            meat  = (T_res * hc3_w[:, None]).T @ T_res
            var_th = M_inv @ meat @ M_inv
            se_vec = np.sqrt(np.maximum(np.diag(var_th), 0))

            for lbl, theta_j, se_j in zip(t_labels, theta_vec, se_vec):
                ci_lo = theta_j - 1.96 * se_j
                ci_hi = theta_j + 1.96 * se_j
                z_j   = theta_j / se_j if se_j > 1e-12 else np.nan
                pval  = float(2 * stats.norm.sf(abs(z_j))) if not np.isnan(z_j) else np.nan
                records.append({**base,
                    "transition":       trans,
                    "transition_label": trans_lbl,
                    "t_component":      lbl,
                    "alt_utils":        alt_used,
                    "theta":      round(float(theta_j), 6),
                    "se":         round(float(se_j), 6),
                    "ci_95":      f"[{ci_lo:.4f}, {ci_hi:.4f}]",
                    "p_value":    round(pval, 4),
                    "reject_H0":  bool(pval < 0.05) if not np.isnan(pval) else "—",
                    "note": (f"θ={theta_j:+.4f}: 1-unit ↑ {lbl} "
                             f"{'↑' if theta_j > 0 else '↓'} P(t{trans}/{trans_lbl}) "
                             f"by {abs(theta_j):.1%}"),
                })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Model visualisation from discovered-model files
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_count(n: int) -> str:
    """Format a count compactly: 12345 → '12.3k'."""
    if n >= 1_000:
        return f"{n/1000:.1f}k"
    return str(n)


def visualize_dfg(model_path: str, min_freq: int = 0) -> "graphviz.Digraph":
    """
    Render a directly-follows graph (.dfg JSON) as a graphviz Digraph.

    Node colour:  green  = start activity
                  coral  = end activity
                  orange = start AND end
                  steel  = normal
    Edge width scales with count; label shows abbreviated count.

    Parameters
    ----------
    model_path : path to a .dfg JSON file
    min_freq   : hide edges whose count is below this threshold (default 0 = show all)
    """
    import json
    try:
        from graphviz import Digraph
    except ImportError:
        raise ImportError("pip install graphviz")

    with open(model_path) as f:
        data = json.load(f)

    activities   = data.get("activities", {})          # {name: count_str}
    dfg_edges    = data.get("directly_follows_relations", [])  # [[[src,dst], count_str], ...]
    start_acts   = set(data.get("start_activities", []))
    end_acts     = set(data.get("end_activities", []))

    # Node colours
    def _fill(act: str) -> str:
        is_start = act in start_acts
        is_end   = act in end_acts
        if is_start and is_end:
            return "#f0b429"   # orange
        if is_start:
            return "#a8d5a2"   # green
        if is_end:
            return "#f4a391"   # coral
        return "#aec6e8"       # steel blue

    dot = Digraph(
        comment="Directly-follows graph",
        graph_attr={"rankdir": "LR", "fontname": "Helvetica"},
        node_attr={"fontname": "Helvetica", "fontsize": "10"},
        edge_attr={"fontname": "Helvetica", "fontsize": "8"},
    )

    for act, cnt_str in activities.items():
        cnt = int(cnt_str)
        dot.node(
            act,
            label=f"{act}\n{_fmt_count(cnt)}",
            shape="box",
            style="filled,rounded",
            fillcolor=_fill(act),
        )

    # Scale edge widths
    counts = [int(c) for (_, c) in dfg_edges]
    max_cnt = max(counts, default=1)

    for (src_dst, cnt_str) in dfg_edges:
        src, dst = src_dst
        cnt = int(cnt_str)
        if cnt < min_freq:
            continue
        width = max(0.4, 3.5 * cnt / max_cnt)
        dot.edge(src, dst,
                 label=_fmt_count(cnt),
                 penwidth=str(round(width, 2)),
                 color="#555555")

    return dot


def visualize_ptree(model_path: str) -> "graphviz.Digraph":
    """
    Render a process tree (.ptree) as a graphviz Digraph.

    All discovered trees in this project are flat flower models
    (loop over all activities).  The visualisation shows the activities
    as a single group with a central loop label, and prints a note.
    """
    try:
        from graphviz import Digraph
    except ImportError:
        raise ImportError("pip install graphviz")

    activities = []
    with open(model_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("activity "):
                activities.append(line[len("activity "):])

    dot = Digraph(
        comment="Process tree (flower model)",
        graph_attr={"rankdir": "LR", "fontname": "Helvetica",
                    "label": "Flower model — any activity sequence is allowed",
                    "labelloc": "t", "fontsize": "12"},
        node_attr={"fontname": "Helvetica", "fontsize": "10"},
    )

    dot.node("_loop_", label="↺ LOOP\n(any order)", shape="diamond",
             style="filled", fillcolor="#ffe0a0")
    for act in activities:
        dot.node(act, label=act, shape="box", style="filled,rounded",
                 fillcolor="#aec6e8")
        dot.edge("_loop_", act, arrowhead="none", color="#aaaaaa")

    return dot


def find_model_file(exs_path: str, models_dir: str) -> str | None:
    """
    Given an exs file path and a directory of model files, return the path
    to the corresponding .dfg or .ptree file, or None if not found.

    Naming convention:
        <stem>-dfg.exs  →  <models_dir>/<stem>-dfg.dfg
        <stem>-flw.exs  →  <models_dir>/<stem>-flw.ptree
    """
    import os
    basename = os.path.basename(exs_path)           # e.g. "bpic12-a.xes.gz-dfg.exs"
    stem     = basename[:-len(".exs")]              # e.g. "bpic12-a.xes.gz-dfg"
    if stem.endswith("-dfg"):
        candidate = os.path.join(models_dir, stem + ".dfg")
    elif stem.endswith("-flw"):
        candidate = os.path.join(models_dir, stem + ".ptree")
    else:
        return None
    return candidate if os.path.exists(candidate) else None


def build_petri_net_graph(df: pd.DataFrame, min_edge_freq: int = 1):
    """
    Reconstruct a transition-level dependency graph from a parse_exs() DataFrame.

    Edges represent causal enabling: if transition B became enabled at step k
    (move_index_of_enablement == k) and transition A fired at step k in the
    same trace (move_index == k), then A → B (A produced the token enabling B).

    Aggregated across all traces this gives the arc structure of the underlying
    Petri net at the transition level, without needing the original model file.

    Node labels use activity_t{tid} format (e.g. A_APPROVED_t8) so that
    multiple transitions sharing the same activity label remain visually distinct.

    Parameters
    ----------
    df            : output of parse_exs()
    min_edge_freq : minimum times A→B must occur to draw the edge

    Returns
    -------
    graphviz.Digraph  (call .view() or .render() to display/save)

    Requires: pip install graphviz
    """
    try:
        from graphviz import Digraph
    except ImportError:
        raise ImportError("pip install graphviz")

    required = {"transition", "move_index", "case"}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame missing columns: {required - set(df.columns)}")
    if "move_index_of_enablement" not in df.columns:
        raise ValueError("DataFrame has no move_index_of_enablement — "
                         "only new-format .exs files contain this field.")

    # Build transition → activity label mapping from visible (non-silent) rows
    t2l: dict = {}
    if "activity" in df.columns:
        t2l = (
            df[df["activity"].notna()]
            .drop_duplicates("transition")[["transition", "activity"]]
            .assign(transition=lambda x: x["transition"].astype(int))
            .set_index("transition")["activity"]
            .astype(str)
            .to_dict()
        )

    def _node_label(tid: int) -> str:
        if tid in t2l:
            return f"{t2l[tid]}_t{tid}"
        return f"τ{tid}"

    # --- Edge counts via vectorised merge (avoids slow iterrows) ---------------

    # Firing index: one row per (case, move_index) → src transition
    firing_df = (
        df[df["move_index"].notna()][["case", "move_index", "transition"]]
        .copy()
        .astype({"case": int, "move_index": int, "transition": int})
        .rename(columns={"transition": "src_tid"})
        .drop_duplicates(subset=["case", "move_index"])   # move_index is unique per trace
    )

    # Enabled-by index: one row per row with a valid moe → dst transition
    moe_df = (
        df[df["move_index_of_enablement"].notna()][
            ["case", "move_index_of_enablement", "transition"]
        ]
        .copy()
        .astype({"case": int, "move_index_of_enablement": int, "transition": int})
        .rename(columns={"transition": "dst_tid"})
    )

    # Join: moe points back to the firing that enabled this transition
    merged = moe_df.merge(
        firing_df,
        left_on=["case", "move_index_of_enablement"],
        right_on=["case", "move_index"],
        how="inner",
    )

    edge_counts = (
        merged.groupby(["src_tid", "dst_tid"], sort=False)
        .size()
        .reset_index(name="count")
    )
    edge_counts = edge_counts[edge_counts["count"] >= min_edge_freq]

    # All nodes: transitions that appear in an edge OR are enabled from the
    # initial marking (moe is null → no predecessor in data)
    involved: set = (
        set(edge_counts["src_tid"].tolist())
        | set(edge_counts["dst_tid"].tolist())
        | set(df[df["move_index_of_enablement"].isna()]["transition"].astype(int).tolist())
    )

    # --- Build graphviz diagram ------------------------------------------------
    dot = Digraph(comment="Transition dependency graph", graph_attr={"rankdir": "LR"})

    for tid in sorted(involved):
        is_silent = tid not in t2l
        dot.node(
            str(tid),
            label=_node_label(tid),
            shape="box",
            style="filled",
            fillcolor="#dddddd" if is_silent else "#aec6e8",
            fontcolor="#555555" if is_silent else "#000000",
        )

    max_freq = int(edge_counts["count"].max()) if not edge_counts.empty else 1
    for _, row in edge_counts.iterrows():
        src, dst, cnt = int(row["src_tid"]), int(row["dst_tid"]), int(row["count"])
        width = max(0.5, 3.0 * cnt / max_freq)
        dot.edge(str(src), str(dst), label=str(cnt),
                 penwidth=str(round(width, 2)), fontsize="9")

    return dot
