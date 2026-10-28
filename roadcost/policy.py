"""Case-level decision rules and their expected cost.

Definitions (used everywhere, by train.py and app.py):

* Ground truth: a case is POSITIVE when its labelled class is Snow or Slush
  (the segment needs treatment now); NEGATIVE when it is Clear or Wet.
* Score: s = P(Snow) + P(Slush) from the classifier's softmax. It is a ranking
  score, not a calibrated probability; thresholds are chosen empirically.
* Single threshold t (model-only policy): FAIL (dispatch treatment) if s >= t, else PASS.
* Two thresholds t_low <= t_high (review policy):
      s <  t_low            -> PASS   (no action)
      t_low <= s < t_high   -> REVIEW (duty officer decides)
      s >= t_high           -> FAIL   (dispatch treatment)
  t_low == t_high means no review band, i.e. the single-threshold policy.
* Expected cost per case uses class-conditional rates and the assumed production
  prevalence pi (never the raw split counts):
      E = pi * [P(FAIL|pos) C_TP + P(PASS|pos) C_FN + P(REV|pos) (c_rev + (1-m) C_TP + m C_FN)]
        + (1-pi) * [P(FAIL|neg) C_FP + P(PASS|neg) C_TN + P(REV|neg) (c_rev + f C_FP + (1-f) C_TN)]
  with m = reviewer miss rate, f = reviewer false-alarm rate, c_rev = review labour per case.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from .config import NEGATIVE_CLASSES, POSITIVE_CLASSES, Economics

EPS_ABOVE_ONE = 1.000001  # a threshold above every possible score => everything PASSes


def is_positive(class_names) -> np.ndarray:
    return np.isin(np.asarray(class_names), POSITIVE_CLASSES).astype(int)


def validate_thresholds(t_low: float, t_high: float) -> None:
    if not (np.isfinite(t_low) and np.isfinite(t_high)):
        raise ValueError("Thresholds must be finite numbers.")
    if t_low < 0 or t_high > EPS_ABOVE_ONE + 1e-9:
        raise ValueError("Thresholds must lie in [0, 1].")
    if t_low > t_high:
        raise ValueError(f"Lower threshold ({t_low:.3f}) cannot exceed upper threshold ({t_high:.3f}).")


def decide(scores, t_low: float, t_high: float) -> np.ndarray:
    """Return an array of 'PASS' / 'REVIEW' / 'FAIL' with the boundary rules above."""
    validate_thresholds(t_low, t_high)
    s = np.asarray(scores, dtype=float)
    out = np.full(s.shape, "REVIEW", dtype=object)
    out[s < t_low] = "PASS"
    out[s >= t_high] = "FAIL"
    return out


def three_way_rates(scores, y, t_low: float, t_high: float) -> dict:
    """Class-conditional shares of PASS/REVIEW/FAIL, plus raw counts."""
    d = decide(scores, t_low, t_high)
    y = np.asarray(y).astype(int)
    pos, neg = y == 1, y == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Need both positive and negative cases to compute rates.")
    counts = {
        f"{dec.lower()}_{grp}": int(((d == dec) & m).sum())
        for dec in ("PASS", "REVIEW", "FAIL")
        for grp, m in (("pos", pos), ("neg", neg))
    }
    rates = {
        "p_fail_pos": counts["fail_pos"] / n_pos,
        "p_pass_pos": counts["pass_pos"] / n_pos,
        "p_rev_pos": counts["review_pos"] / n_pos,
        "p_fail_neg": counts["fail_neg"] / n_neg,
        "p_pass_neg": counts["pass_neg"] / n_neg,
        "p_rev_neg": counts["review_neg"] / n_neg,
    }
    return {**rates, "counts": counts, "n_pos": n_pos, "n_neg": n_neg}


def expected_cost(r: dict, e: Economics) -> float:
    """Expected euro cost per case for a three-way rate dict (see module docstring)."""
    m, f, c_rev = e.reviewer_miss_rate, e.reviewer_false_alarm_rate, e.review_cost_per_case
    pos = (r["p_fail_pos"] * e.c_tp + r["p_pass_pos"] * e.c_fn
           + r["p_rev_pos"] * (c_rev + (1 - m) * e.c_tp + m * e.c_fn))
    neg = (r["p_fail_neg"] * e.c_fp + r["p_pass_neg"] * e.c_tn
           + r["p_rev_neg"] * (c_rev + f * e.c_fp + (1 - f) * e.c_tn))
    return e.prevalence * pos + (1 - e.prevalence) * neg


def review_share(r: dict, e: Economics) -> float:
    """Production-weighted share of cases sent to review."""
    return e.prevalence * r["p_rev_pos"] + (1 - e.prevalence) * r["p_rev_neg"]


def staffing(share: float, e: Economics) -> dict:
    cases = share * e.cases_per_day
    hours = cases * e.review_seconds / 3600.0
    fte = hours / e.productive_hours_per_reviewer_day
    return {
        "review_share": share,
        "review_cases_per_day": cases,
        "review_hours_per_day": hours,
        "reviewer_fte": fte,
        "whole_reviewers_needed": int(math.ceil(fte - 1e-9)) if fte > 0 else 0,
        "fits_spare_capacity": hours <= e.review_capacity_hours_per_day + 1e-9,
    }


def policy_summary(name: str, r: dict, e: Economics) -> dict:
    """All headline numbers for one policy under one set of economics."""
    cost = expected_cost(r, e)
    share = review_share(r, e)
    pi, n = e.prevalence, e.cases_per_day
    m, f = e.reviewer_miss_rate, e.reviewer_false_alarm_rate
    out = {
        "policy": name,
        "cost_per_case": cost,
        "cost_per_1000": 1000 * cost,
        "cost_per_day": n * cost,
        # automated (model-made) errors per day
        "auto_fp_per_day": n * (1 - pi) * r["p_fail_neg"],
        "auto_fn_per_day": n * pi * r["p_pass_pos"],
        # residual errors made by reviewers on deferred cases
        "review_fp_per_day": n * (1 - pi) * r["p_rev_neg"] * f,
        "review_fn_per_day": n * pi * r["p_rev_pos"] * m,
        "review_labour_per_day": n * share * e.review_cost_per_case,
        **{k: v for k, v in r.items() if k.startswith("p_")},
    }
    out["total_fp_per_day"] = out["auto_fp_per_day"] + out["review_fp_per_day"]
    out["total_fn_per_day"] = out["auto_fn_per_day"] + out["review_fn_per_day"]
    out.update(staffing(share, e))
    return out


# ---------------------------------------------------------------- fixed reference policies
def constant_rates(decision: str) -> dict:
    """Rates for a policy that ignores the image (PASS all, FAIL all or REVIEW all)."""
    r = {k: 0.0 for k in ("p_fail_pos", "p_pass_pos", "p_rev_pos", "p_fail_neg", "p_pass_neg", "p_rev_neg")}
    key = {"PASS": "pass", "FAIL": "fail", "REVIEW": "rev"}[decision]
    r[f"p_{key}_pos"] = 1.0
    r[f"p_{key}_neg"] = 1.0
    return r


# ---------------------------------------------------------------- threshold sweeps
def candidate_thresholds(scores, max_points: int = 400) -> np.ndarray:
    """Midpoints between sorted unique scores (subsampled), plus 0 and >1.

    Every distinct operating point of the empirical ROC is reachable this way, so the
    sweep does not depend on an arbitrary linear grid.
    """
    u = np.unique(np.asarray(scores, dtype=float))
    mids = (u[:-1] + u[1:]) / 2 if len(u) > 1 else u
    if len(mids) > max_points:
        idx = np.unique(np.linspace(0, len(mids) - 1, max_points).round().astype(int))
        mids = mids[idx]
    grid = np.concatenate([[0.0], mids, np.linspace(0, 1, 101)[1:-1], [EPS_ABOVE_ONE]])
    return np.unique(grid)


def _below(scores, grid):
    """F(t) = share of scores strictly below t, for every t in grid."""
    s = np.sort(np.asarray(scores, dtype=float))
    return np.searchsorted(s, grid, side="left") / max(len(s), 1)


def sweep_single(scores, y, grid, e: Economics) -> dict:
    """Expected cost of the model-only policy at every threshold in grid."""
    y = np.asarray(y).astype(int)
    s = np.asarray(scores, dtype=float)
    fp_below = _below(s[y == 1], grid)  # P(PASS | pos) = FNR
    fn_below = _below(s[y == 0], grid)  # P(PASS | neg) = TNR
    tpr, fnr = 1 - fp_below, fp_below
    tnr, fpr = fn_below, 1 - fn_below
    cost = (e.prevalence * (tpr * e.c_tp + fnr * e.c_fn)
            + (1 - e.prevalence) * (fpr * e.c_fp + tnr * e.c_tn))
    return {"threshold": np.asarray(grid), "tpr": tpr, "fpr": fpr, "cost_per_case": cost}


def best_single(sweep: dict) -> tuple[float, float]:
    """Lowest-cost threshold; ties broken towards the LOWER threshold (more treatment, safer)."""
    c = sweep["cost_per_case"]
    i = int(np.flatnonzero(np.isclose(c, c.min(), rtol=0, atol=1e-9))[0])
    return float(sweep["threshold"][i]), float(c[i])


def sweep_pairs(scores, y, grid, e: Economics) -> dict:
    """Expected cost and review share for every (t_low, t_high) pair with t_low <= t_high."""
    y = np.asarray(y).astype(int)
    s = np.asarray(scores, dtype=float)
    g = np.asarray(grid)
    Fp, Fn = _below(s[y == 1], g), _below(s[y == 0], g)
    L, H = np.meshgrid(np.arange(len(g)), np.arange(len(g)), indexing="ij")
    valid = L <= H
    p_pass_pos, p_fail_pos = Fp[L], 1 - Fp[H]
    p_rev_pos = Fp[H] - Fp[L]
    p_pass_neg, p_fail_neg = Fn[L], 1 - Fn[H]
    p_rev_neg = Fn[H] - Fn[L]
    m, f, c_rev = e.reviewer_miss_rate, e.reviewer_false_alarm_rate, e.review_cost_per_case
    pos = p_fail_pos * e.c_tp + p_pass_pos * e.c_fn + p_rev_pos * (c_rev + (1 - m) * e.c_tp + m * e.c_fn)
    neg = p_fail_neg * e.c_fp + p_pass_neg * e.c_tn + p_rev_neg * (c_rev + f * e.c_fp + (1 - f) * e.c_tn)
    cost = e.prevalence * pos + (1 - e.prevalence) * neg
    share = e.prevalence * p_rev_pos + (1 - e.prevalence) * p_rev_neg
    cost = np.where(valid, cost, np.inf)
    return {"grid": g, "cost": cost, "share": np.where(valid, share, np.nan)}


def best_pair(pairs: dict, e: Economics, capacity_hours: float | None = None) -> dict:
    """Lowest-cost pair, optionally subject to review hours/day <= capacity_hours.

    Ties: prefer the smaller review share, then the lower t_low.
    """
    cost = pairs["cost"].copy()
    hours = pairs["share"] * e.cases_per_day * e.review_seconds / 3600.0
    if capacity_hours is not None:
        cost = np.where(hours <= capacity_hours + 1e-9, cost, np.inf)
    best = np.nanmin(cost)
    cand = np.argwhere(np.isclose(cost, best, rtol=0, atol=1e-9))
    cand = sorted(cand.tolist(), key=lambda ij: (pairs["share"][ij[0], ij[1]], ij[0], ij[1]))
    i, j = cand[0]
    g = pairs["grid"]
    return {"t_low": float(g[i]), "t_high": float(g[j]), "cost_per_case": float(cost[i, j]),
            "review_share": float(pairs["share"][i, j])}


# ---------------------------------------------------------------- uncertainty
def bootstrap_cost(scores, y, t_low, t_high, e: Economics, n_boot: int = 2000, seed: int = 0) -> dict:
    """Percentile CI for expected cost, resampling positives and negatives separately.

    Prevalence is an external assumption, so only the class-conditional rates are
    resampled (stratified bootstrap). This captures sampling noise in the evaluation
    split, not uncertainty in the cost assumptions or in production drift.
    """
    rng = np.random.default_rng(seed)
    s = np.asarray(scores, dtype=float)
    y = np.asarray(y).astype(int)
    sp, sn = s[y == 1], s[y == 0]
    vals = np.empty(n_boot)
    for b in range(n_boot):
        bp = rng.choice(sp, len(sp), replace=True)
        bn = rng.choice(sn, len(sn), replace=True)
        bs = np.concatenate([bp, bn])
        by = np.concatenate([np.ones(len(bp), int), np.zeros(len(bn), int)])
        vals[b] = expected_cost(three_way_rates(bs, by, t_low, t_high), e)
    return {"mean": float(vals.mean()), "lo95": float(np.percentile(vals, 2.5)),
            "hi95": float(np.percentile(vals, 97.5))}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def with_overrides(e: Economics, **kw) -> Economics:
    return replace(e, **kw)


__all__ = [n for n in dir() if not n.startswith("_")] + ["NEGATIVE_CLASSES", "POSITIVE_CLASSES"]
