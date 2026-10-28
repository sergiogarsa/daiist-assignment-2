"""
Assignment 2 dashboard (run via `uv run python main.py app`).

Loads ONLY what `main.py train` already produced: weights/best.pt and artifacts/.
It never trains or downloads a model. Interactive policy numbers are recomputed from
the saved per-image scores, with the same cost code (roadcost.policy) used in training.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from roadcost import policy as P
from roadcost.config import (
    ART_DIR, ASSUMPTION_NOTES, CLASSES, ECON, POSITIVE_CLASSES, RF_PROJECT, RF_URL, RF_VERSION,
    RF_WORKSPACE, ROOT, WEIGHTS_PATH, Economics,
)

# ----------------------------------------------------------------------------- loading
LOAD_ERRORS: list[str] = []


def _load_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception as e:  # noqa: BLE001
        LOAD_ERRORS.append(f"Could not read {p.relative_to(ROOT)}: {e}. Run `uv run python main.py train`.")
        return None


def _load_csv(p: Path):
    try:
        return pd.read_csv(p)
    except Exception as e:  # noqa: BLE001
        LOAD_ERRORS.append(f"Could not read {p.relative_to(ROOT)}: {e}. Run `uv run python main.py train`.")
        return None


MODEL = None
if WEIGHTS_PATH.is_file():
    try:
        from ultralytics import YOLO

        MODEL = YOLO(str(WEIGHTS_PATH))
    except Exception as e:  # noqa: BLE001
        LOAD_ERRORS.append(f"weights/best.pt exists but failed to load: {e}")
else:
    LOAD_ERRORS.append("weights/best.pt is missing. Image upload is disabled until `main.py train` has run.")

RESULTS = _load_json(ART_DIR / "results.json") or {}
AUDIT = _load_json(ART_DIR / "dataset_audit.json") or {}
PRED = {"valid": _load_csv(ART_DIR / "predictions_valid.csv"), "test": _load_csv(ART_DIR / "predictions_test.csv")}
SEL = RESULTS.get("selection", {})
T_STAR = float(SEL.get("t_star", 0.5))
PAIR = SEL.get("pair_constrained", {"t_low": 0.3, "t_high": 0.7})
T_LOW0, T_HIGH0 = float(PAIR["t_low"]), float(PAIR["t_high"])
DATA_OK = PRED["valid"] is not None and PRED["test"] is not None


def econ_from(c_fp, c_fn, prev, cases, rev_s, hourly, prod_h, cap_h, miss, fa) -> Economics:
    vals = dict(c_fp=c_fp, c_fn=c_fn, prevalence=prev, cases_per_day=cases, review_seconds=rev_s,
                reviewer_hourly_cost=hourly, productive_hours_per_reviewer_day=prod_h,
                review_capacity_hours_per_day=cap_h, reviewer_miss_rate=miss, reviewer_false_alarm_rate=fa)
    for k, v in vals.items():
        if v is None or not np.isfinite(float(v)):
            raise ValueError(f"{k} must be a number.")
    if not 0 < prev < 1:
        raise ValueError("Prevalence must be strictly between 0 and 1.")
    if min(c_fp, c_fn, cases, rev_s, hourly) < 0 or prod_h <= 0:
        raise ValueError("Costs, volumes and times must be non-negative (productive hours > 0).")
    if not (0 <= miss <= 1 and 0 <= fa <= 1):
        raise ValueError("Reviewer error rates must be in [0, 1].")
    return Economics(c_fp=float(c_fp), c_fn=float(c_fn), prevalence=float(prev), cases_per_day=int(cases),
                     review_seconds=float(rev_s), reviewer_hourly_cost=float(hourly),
                     productive_hours_per_reviewer_day=float(prod_h), review_capacity_hours_per_day=float(cap_h),
                     reviewer_miss_rate=float(miss), reviewer_false_alarm_rate=float(fa))


# ----------------------------------------------------------------------------- inference tab
def classify(img, t_single, t_low, t_high):
    if MODEL is None:
        raise gr.Error("No trained weights loaded (weights/best.pt). Run `uv run python main.py train` first.")
    if img is None:
        raise gr.Error("Please upload an image (JPG/PNG).")
    try:
        P.validate_thresholds(t_low, t_high)
        P.validate_thresholds(t_single, t_single)
    except ValueError as e:
        raise gr.Error(str(e)) from e
    if min(img.size) < 32:
        raise gr.Error("Image is too small to classify (need at least 32x32 pixels).")
    try:
        res = MODEL.predict(img.convert("RGB"), imgsz=224, device="cpu", verbose=False)[0]
    except Exception as e:  # noqa: BLE001
        raise gr.Error(f"Could not run the model on this image: {e}") from e
    probs = res.probs.data.cpu().numpy()
    names = [res.names[i] for i in range(len(probs))]
    label = {n: float(p) for n, p in zip(names, probs)}
    s = float(sum(label[c] for c in POSITIVE_CLASSES))
    single = "FAIL - dispatch treatment" if s >= t_single else "PASS - no action"
    three = P.decide([s], t_low, t_high)[0]
    three_txt = {"PASS": "PASS - no action", "REVIEW": "HUMAN REVIEW - duty officer decides",
                 "FAIL": "FAIL - dispatch treatment"}[three]
    md = (f"**Treatment score** s = P(Snow) + P(Slush) = **{s:.4f}**  \n"
          f"**Model-only policy** (FAIL if s >= {t_single:.3f}): **{single}**  \n"
          f"**Review policy** (PASS if s < {t_low:.3f}; REVIEW if {t_low:.3f} <= s < {t_high:.3f}; "
          f"FAIL if s >= {t_high:.3f}): **{three_txt}**  \n"
          f"_Scores are softmax outputs, not calibrated probabilities. The model was trained on in-vehicle "
          f"camera images from Utah; other camera types may behave differently._")
    return label, md


# ----------------------------------------------------------------------------- policy tab
def _rates(split, t_low, t_high):
    d = PRED[split]
    return P.three_way_rates(d["score"].to_numpy(), d["y"].to_numpy(), t_low, t_high)


def cost_curve_fig(split, e: Economics, t_single):
    fig = go.Figure()
    for sp, dash in (("valid", "solid"), ("test", "dash")):
        d = PRED[sp]
        grid = P.candidate_thresholds(d["score"])
        sw = P.sweep_single(d["score"], d["y"], grid, e)
        fig.add_trace(go.Scatter(x=sw["threshold"], y=1000 * sw["cost_per_case"], mode="lines",
                                 name=f"{sp}{' (selection)' if sp == 'valid' else ' (exploratory)'}",
                                 line={"dash": dash}))
    sw_v = P.sweep_single(PRED["valid"]["score"], PRED["valid"]["y"], P.candidate_thresholds(PRED["valid"]["score"]), e)
    t_now, c_now = P.best_single(sw_v)
    fig.add_vline(x=T_STAR, line_color="red", annotation_text=f"frozen t* = {T_STAR:.3f}", annotation_position="top")
    fig.add_vline(x=t_single, line_color="orange", line_dash="dot", annotation_text=f"slider t = {t_single:.3f}",
                  annotation_position="bottom")
    fig.add_trace(go.Scatter(x=[t_now], y=[1000 * c_now], mode="markers", marker={"size": 10, "color": "purple"},
                             name=f"valid minimum under current assumptions (t={t_now:.3f})"))
    fig.add_hline(y=1000 * P.expected_cost(P.constant_rates("PASS"), e), line_dash="dot", line_color="grey",
                  annotation_text="do nothing")
    fig.update_layout(height=380, margin=dict(l=40, r=20, t=30, b=40),
                      xaxis_title="threshold t on s (FAIL if s >= t)", yaxis_title="EUR per 1,000 cases",
                      legend=dict(orientation="h", y=-0.25))
    return fig


def update_policy(split, t_single, t_low, t_high, c_fp, c_fn, prev, cases, rev_s, hourly, prod_h, cap_h, miss, fa):
    if not DATA_OK:
        raise gr.Error("Saved evaluation outputs are missing. Run `uv run python main.py train`.")
    try:
        e = econ_from(c_fp, c_fn, prev, cases, rev_s, hourly, prod_h, cap_h, miss, fa)
        P.validate_thresholds(t_low, t_high)
        P.validate_thresholds(t_single, t_single)
    except ValueError as err:
        raise gr.Error(str(err)) from err

    r1 = _rates(split, t_single, t_single)
    r2 = _rates(split, t_low, t_high)
    c1, c2 = r1["counts"], r2["counts"]
    n_pos, n_neg = r1["n_pos"], r1["n_neg"]
    cm_single = pd.DataFrame(
        {"model PASS (no action)": [c1["pass_pos"], c1["pass_neg"]],
         "model FAIL (dispatch)": [c1["fail_pos"], c1["fail_neg"]]},
        index=[f"true positive: Snow/Slush (n={n_pos})", f"true negative: Clear/Wet (n={n_neg})"])
    cm_review = pd.DataFrame(
        {"auto PASS": [c2["pass_pos"], c2["pass_neg"]], "sent to REVIEW": [c2["review_pos"], c2["review_neg"]],
         "auto FAIL": [c2["fail_pos"], c2["fail_neg"]]},
        index=[f"true positive: Snow/Slush (n={n_pos})", f"true negative: Clear/Wet (n={n_neg})"])
    # expected outcome of reviewed cases under the stated reviewer error rates (NOT measured)
    post = pd.DataFrame(
        {"final PASS": [c2["pass_pos"] + c2["review_pos"] * e.reviewer_miss_rate,
                        c2["pass_neg"] + c2["review_neg"] * (1 - e.reviewer_false_alarm_rate)],
         "final FAIL": [c2["fail_pos"] + c2["review_pos"] * (1 - e.reviewer_miss_rate),
                        c2["fail_neg"] + c2["review_neg"] * e.reviewer_false_alarm_rate]},
        index=cm_review.index).round(1)
    cm_single.index.name = cm_review.index.name = post.index.name = f"{split} split"

    pols = [
        P.policy_summary("1. Do nothing (no camera-based dispatch)", P.constant_rates("PASS"), e),
        P.policy_summary("2. Human reviews every case", P.constant_rates("REVIEW"), e),
        P.policy_summary(f"3. Model only, t = {t_single:.3f}", r1, e),
        P.policy_summary(f"4. Model + review, [{t_low:.3f}, {t_high:.3f})", r2, e),
        P.policy_summary("(ref) Treat every segment", P.constant_rates("FAIL"), e),
    ]
    tab = pd.DataFrame([{
        "policy": p["policy"], "EUR / 1,000 cases": round(p["cost_per_1000"], 1),
        "EUR / day": round(p["cost_per_day"], 0), "review share": f"{p['review_share']:.1%}",
        "review cases/day": round(p["review_cases_per_day"], 1), "review h/day": round(p["review_hours_per_day"], 2),
        "reviewer FTE": round(p["reviewer_fte"], 2), "whole reviewers": p["whole_reviewers_needed"],
        "auto FP/day": round(p["auto_fp_per_day"], 1), "auto FN/day": round(p["auto_fn_per_day"], 1),
        "reviewer FP+FN/day": round(p["review_fp_per_day"] + p["review_fn_per_day"], 1),
    } for p in pols])
    p4 = pols[3]
    fits = "fits within" if p4["fits_spare_capacity"] else "EXCEEDS"
    warn = ("\n\n> **Exploratory view of the TEST split.** The frozen policy below was chosen on VALID. "
            "Any thresholds you pick here after looking at test results are not an unbiased final policy."
            if split == "test" else "")
    md = (f"### Review policy at the current sliders ({split} rates, production-weighted)\n"
          f"- Expected cost: **EUR {p4['cost_per_1000']:.1f} per 1,000 cases**, EUR {p4['cost_per_day']:.0f} per day "
          f"at {e.cases_per_day} cases/day\n"
          f"- Sent to review: **{p4['review_share']:.1%}** = {p4['review_cases_per_day']:.1f} cases/day = "
          f"{p4['review_hours_per_day']:.2f} h/day = {p4['reviewer_fte']:.2f} FTE "
          f"({p4['whole_reviewers_needed']} whole reviewer(s) at {e.productive_hours_per_reviewer_day:g} productive h/day); "
          f"this **{fits}** the {e.review_capacity_hours_per_day:g} h/day of spare duty-officer capacity\n"
          f"- Remaining automated errors: {p4['auto_fp_per_day']:.1f} false dispatches/day, "
          f"{p4['auto_fn_per_day']:.1f} missed snowy segments/day\n"
          f"- Model only at t = {t_single:.3f}: EUR {pols[2]['cost_per_1000']:.1f} per 1,000 cases"
          f"{warn}")
    return cm_single, cm_review, post, tab, md, cost_curve_fig(split, e, t_single)


def reset_frozen():
    return T_STAR, T_LOW0, T_HIGH0


# ----------------------------------------------------------------------------- static views
def per_class_table():
    rows = []
    for split in ("valid", "test"):
        mc = RESULTS.get("multiclass", {}).get(split)
        if not mc:
            continue
        for c in CLASSES:
            v = mc["per_class"][c]
            rows.append({"split": split, "class": c, "group": "positive" if c in POSITIVE_CLASSES else "negative",
                         "precision": round(v["precision"], 3), "recall": round(v["recall"], 3),
                         "F1": round(v["f1"], 3), "support": v["support"]})
        rows.append({"split": split, "class": "OVERALL accuracy", "group": "", "precision": None,
                     "recall": round(mc["accuracy"], 3), "F1": round(mc["macro_f1"], 3), "support": None})
    return pd.DataFrame(rows)


def confusion_fig(split):
    mc = RESULTS.get("multiclass", {}).get(split)
    if not mc:
        return go.Figure()
    cm = np.asarray(mc["confusion"])
    norm = cm / cm.sum(axis=1, keepdims=True)
    text = [[f"{cm[i, j]}<br>{norm[i, j]:.0%}" for j in range(4)] for i in range(4)]
    fig = go.Figure(go.Heatmap(z=norm, x=[f"pred {c}" for c in CLASSES], y=[f"true {c}" for c in CLASSES],
                               text=text, texttemplate="%{text}", colorscale="Blues", zmin=0, zmax=1))
    fig.update_layout(height=360, margin=dict(l=40, r=20, t=30, b=40), yaxis_autorange="reversed",
                      title=f"Multiclass confusion matrix ({split}), row-normalised")
    return fig


def baseline_table():
    if not RESULTS:
        return pd.DataFrame()
    pt = {p["policy"]: p for p in RESULTS["policies_test"]}
    cm = RESULTS["case_metrics"]
    mo = [p for p in RESULTS["policies_test"] if p["policy"].startswith("Model only")][0]
    zs = cm["zero_shot_imagenet"]["test_policy"]
    rows = [
        {"model / baseline (TEST split)": "Fine-tuned yolo11n-cls", "4-class accuracy": RESULTS["multiclass"]["test"]["accuracy"],
         "case AUC": cm["fine_tuned"]["auc_test"], "TPR": mo["p_fail_pos"], "FPR": mo["p_fail_neg"],
         "EUR / 1,000": mo["cost_per_1000"]},
        {"model / baseline (TEST split)": "Zero-shot ImageNet yolo11n-cls (snow-class mapping)", "4-class accuracy": None,
         "case AUC": cm["zero_shot_imagenet"]["auc_test"], "TPR": zs["p_fail_pos"], "FPR": zs["p_fail_neg"],
         "EUR / 1,000": zs["cost_per_1000"]},
        {"model / baseline (TEST split)": "Zero-shot COCO yolo11n (skis/snowboard box => FAIL; no road-surface class)",
         "4-class accuracy": None, "case AUC": RESULTS["coco_zero_shot"].get("auc_test_winter_score"),
         "TPR": RESULTS["coco_zero_shot"]["test_policy"]["p_fail_pos"],
         "FPR": RESULTS["coco_zero_shot"]["test_policy"]["p_fail_neg"],
         "EUR / 1,000": RESULTS["coco_zero_shot"]["test_policy"]["cost_per_1000"]},
        {"model / baseline (TEST split)": f"Majority class ({RESULTS['majority_baseline']['class']}) => always FAIL",
         "4-class accuracy": RESULTS["majority_baseline"]["accuracy"], "case AUC": 0.5, "TPR": 1.0, "FPR": 1.0,
         "EUR / 1,000": pt["Treat every segment (always dispatch)"]["cost_per_1000"]},
    ]
    df = pd.DataFrame(rows)
    for c in ("4-class accuracy", "case AUC", "TPR", "FPR"):
        df[c] = df[c].map(lambda v: None if v is None or pd.isna(v) else round(float(v), 3))
    df["EUR / 1,000"] = df["EUR / 1,000"].round(1)
    return df


def assumptions_md():
    e = ECON
    lines = ["### Economic assumptions (baseline, used for the frozen policy)",
             "| item | value | basis |", "|---|---|---|"]
    items = [("C_FP: wasted treatment sortie", f"EUR {e.c_fp:,.0f}", ASSUMPTION_NOTES["c_fp"]),
             ("C_FN: snow/slush left untreated", f"EUR {e.c_fn:,.0f}", ASSUMPTION_NOTES["c_fn"]),
             ("C_TP, C_TN", "EUR 0", "Regret accounting: necessary treatment is paid under every policy."),
             ("production prevalence pi", f"{e.prevalence:.0%}", ASSUMPTION_NOTES["prevalence"]),
             ("cases per day", f"{e.cases_per_day:,}", ASSUMPTION_NOTES["cases_per_day"]),
             ("review time per case", f"{e.review_seconds:g} s", ASSUMPTION_NOTES["review_seconds"]),
             ("reviewer cost", f"EUR {e.reviewer_hourly_cost:g}/h => EUR {e.review_cost_per_case:.2f}/case",
              ASSUMPTION_NOTES["reviewer_hourly_cost"]),
             ("productive review h per reviewer-day", f"{e.productive_hours_per_reviewer_day:g}",
              ASSUMPTION_NOTES["productive_hours_per_reviewer_day"]),
             ("spare review capacity", f"{e.review_capacity_hours_per_day:g} h/day",
              ASSUMPTION_NOTES["review_capacity_hours_per_day"]),
             ("reviewer miss / false alarm", f"{e.reviewer_miss_rate:.0%} / {e.reviewer_false_alarm_rate:.0%}",
              ASSUMPTION_NOTES["reviewer_error"])]
    for a, b, c in items:
        lines.append(f"| {a} | {b} | {c} |")
    lines += ["", "### Policy definitions",
              "- **Case** = one camera snapshot of one road segment at one decision cycle. "
              "**Positive** = labelled Snow or Slush (needs treatment now); **negative** = Clear or Wet.",
              "- **Score** s = P(Snow) + P(Slush). Not a calibrated probability; thresholds are chosen empirically.",
              "- **Model-only (single threshold t):** FAIL (dispatch) if s >= t, else PASS.",
              "- **Model + review (t_low <= t_high):** PASS if s < t_low; HUMAN REVIEW if t_low <= s < t_high; "
              "FAIL if s >= t_high. t_low = t_high means no review band.",
              "- **Do nothing:** never dispatch on camera evidence; snowy segments are treated only when a "
              "patrol/public report finds them later (cost = every positive becomes a FN).",
              "- **Review all:** the duty officer judges every image (pays review labour + reviewer errors).",
              "- Expected cost per case = pi [P(FAIL|pos) C_TP + P(PASS|pos) C_FN + P(REV|pos)(c_rev + m C_FN)] + "
              "(1 - pi) [P(FAIL|neg) C_FP + P(PASS|neg) C_TN + P(REV|neg)(c_rev + f C_FP)], using class-conditional "
              "rates from the split and the ASSUMED production prevalence pi, never raw split counts.",
              "",
              f"### Frozen policy (selected on VALID, reported on TEST)\n- Model-only threshold **t* = {T_STAR:.4f}**\n"
              f"- Review band **[{T_LOW0:.4f}, {T_HIGH0:.4f})** (chosen within {ECON.review_capacity_hours_per_day:g} h/day review capacity)"]
    return "\n".join(lines)


def policy_test_table():
    if not RESULTS:
        return pd.DataFrame()
    return pd.DataFrame([{"policy (frozen, TEST)": p["policy"], "EUR / 1,000": round(p["cost_per_1000"], 1),
                          "EUR / day": round(p["cost_per_day"], 0), "review share": f"{p['review_share']:.1%}",
                          "review h/day": round(p["review_hours_per_day"], 2), "FTE": round(p["reviewer_fte"], 2),
                          "auto FP/day": round(p["auto_fp_per_day"], 1), "auto FN/day": round(p["auto_fn_per_day"], 1)}
                         for p in RESULTS["policies_test"]])


def dataset_md():
    if not AUDIT:
        return "Dataset audit missing."
    c = AUDIT["counts"]
    lines = [f"**Dataset**: [{RF_WORKSPACE}/{RF_PROJECT} v{RF_VERSION}]({RF_URL}) (classification, CC BY 4.0)", "",
             "| class | train | valid | test |", "|---|---|---|---|"]
    for cl in CLASSES:
        lines.append(f"| {cl} | {c['train'][cl]} | {c['valid'][cl]} | {c['test'][cl]} |")
    nd = AUDIT["near_duplicates"]
    lines += ["", f"Snow/Slush share: " + ", ".join(f"{k} {v:.1%}" for k, v in AUDIT["positive_share_by_split"].items()),
              "", "**Near-duplicates of training images (256-bit dHash)**", "",
              "| max Hamming bits | valid share | test share |", "|---|---|---|"]
    for r in nd.get("curve", []):
        lines.append(f"| {r['bits']} | {r['valid_share']:.1%} | {r['test_share']:.1%} |")
    lines += ["", "Bounding-box statistics do not apply: this is a classification dataset."]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- UI
with gr.Blocks(title="Winter road treatment: pass / review / fail") as demo:
    gr.Markdown("# Winter road treatment decision dashboard\n"
                "Should the duty officer dispatch a spot treatment (grit/plough) to this road segment? "
                "Positive = **Snow or Slush** on the road surface. Fine-tuned yolo11n-cls, "
                f"[UDOT Winter Road Conditions v1]({RF_URL}).")
    if LOAD_ERRORS:
        gr.Markdown("\n".join(f"> **Error:** {m}" for m in LOAD_ERRORS))

    with gr.Row():
        t_single = gr.Slider(0, 1, value=T_STAR, step=0.001, label="Model-only threshold t (FAIL if s >= t)")
        t_low = gr.Slider(0, 1, value=T_LOW0, step=0.001, label="t_low (PASS if s < t_low)")
        t_high = gr.Slider(0, 1, value=T_HIGH0, step=0.001, label="t_high (FAIL if s >= t_high)")
        reset = gr.Button("Reset to frozen validation-selected policy")

    with gr.Tab("1. Classify an image"):
        with gr.Row():
            img = gr.Image(type="pil", label="Upload a road camera image")
            with gr.Column():
                probs = gr.Label(num_top_classes=4, label="Class probabilities")
                decision = gr.Markdown()
        btn = gr.Button("Classify", variant="primary")
        btn.click(classify, [img, t_single, t_low, t_high], [probs, decision])
        img.upload(classify, [img, t_single, t_low, t_high], [probs, decision])

    with gr.Tab("2. Policy, costs and staffing"):
        with gr.Row():
            split = gr.Radio(["valid", "test"], value="valid",
                             label="Rates from split (valid = selection; test = exploratory)")
        with gr.Accordion("What-if economic assumptions (baseline values pre-filled)", open=False):
            with gr.Row():
                c_fp = gr.Number(ECON.c_fp, label="C_FP EUR (wasted sortie)")
                c_fn = gr.Number(ECON.c_fn, label="C_FN EUR (untreated snow/slush)")
                prev = gr.Number(ECON.prevalence, label="production prevalence (0-1)")
                cases = gr.Number(ECON.cases_per_day, label="cases per day", precision=0)
            with gr.Row():
                rev_s = gr.Number(ECON.review_seconds, label="review seconds per case")
                hourly = gr.Number(ECON.reviewer_hourly_cost, label="reviewer EUR per hour")
                prod_h = gr.Number(ECON.productive_hours_per_reviewer_day, label="productive h per reviewer-day")
                cap_h = gr.Number(ECON.review_capacity_hours_per_day, label="spare review capacity h/day")
                miss = gr.Number(ECON.reviewer_miss_rate, label="reviewer miss rate")
                fa = gr.Number(ECON.reviewer_false_alarm_rate, label="reviewer false-alarm rate")
        summary = gr.Markdown()
        with gr.Row():
            cm1 = gr.Dataframe(label="Case-level confusion matrix: MODEL ONLY at t (counts)", interactive=False)
            cm2 = gr.Dataframe(label="Case-level decisions: MODEL + REVIEW (deferred cases shown separately)",
                               interactive=False)
            cm3 = gr.Dataframe(label="Expected final outcome after review (ASSUMED reviewer error rates; not measured)",
                               interactive=False)
        comp = gr.Dataframe(label="Policy comparison (same costs, same prevalence, same split rates)", interactive=False)
        curve = gr.Plot(label="Expected cost vs threshold (model-only)")
        inputs = [split, t_single, t_low, t_high, c_fp, c_fn, prev, cases, rev_s, hourly, prod_h, cap_h, miss, fa]
        outputs = [cm1, cm2, cm3, comp, summary, curve]
        for comp_in in inputs:
            comp_in.change(update_policy, inputs, outputs)
        demo.load(update_policy, inputs, outputs)

    with gr.Tab("3. Per-class results and baselines"):
        gr.Dataframe(per_class_table(), label="Per-class precision / recall (fine-tuned model)", interactive=False)
        with gr.Row():
            gr.Plot(confusion_fig("valid"))
            gr.Plot(confusion_fig("test"))
        gr.Dataframe(baseline_table(), label="Baselines on the same TEST split (frozen thresholds from VALID)",
                     interactive=False)
        gr.Dataframe(policy_test_table(), label="Frozen policies evaluated on TEST at baseline assumptions",
                     interactive=False)
        imgs = [str(ART_DIR / f) for f in ("error_examples.png", "training_curves.png", "sensitivity.png",
                                           "policy_map.png", "review_pair_heatmap.png") if (ART_DIR / f).is_file()]
        if imgs:
            gr.Gallery(imgs, label="Error examples, training curves, sensitivity", columns=2, height="auto")

    with gr.Tab("4. Assumptions, definitions, dataset"):
        gr.Markdown(assumptions_md())
        gr.Markdown(dataset_md())

    reset.click(reset_frozen, None, [t_single, t_low, t_high])


if __name__ == "__main__":
    demo.launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
                server_port=int(os.environ.get("APP_PORT", "7860")), show_error=True)
