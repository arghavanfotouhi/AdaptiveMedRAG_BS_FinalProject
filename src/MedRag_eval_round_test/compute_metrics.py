# #!/usr/bin/env python3
# """
# Compute evaluation metrics for Adaptive-k vs. Fixed-k (baseline)
# from eval_round_*.json result files.

# Usage:
#     python compute_metrics.py "eval_round_test*.json"
#     python compute_metrics.py f1.json f2.json ... f10.json
#     python compute_metrics.py ./results_dir
#     python compute_metrics.py "eval_round_*.json" --baseline-k 32 --out report_out

# Outputs (written to the --out directory):
#     summary.csv          : per-file metrics + mean/std across files + pooled
#     k_distribution.csv   : distribution of selected k (count and %)
#     stop_reasons.csv     : distribution of stop reasons (count and %)
#     metrics.json         : all numbers in JSON format
# """
# import argparse
# import csv
# import glob
# import json
# import math
# import os
# import statistics as st
# from collections import Counter


# # ----------------------------------------------------------------------------
# # Loading
# # ----------------------------------------------------------------------------
# def expand_inputs(items):
#     paths = []
#     for it in items:
#         if os.path.isdir(it):
#             paths += sorted(glob.glob(os.path.join(it, "*.json")))
#         else:
#             hits = sorted(glob.glob(it))
#             paths += hits if hits else [it]
#     seen, out = set(), []
#     for p in paths:
#         if p not in seen:
#             seen.add(p)
#             out.append(p)
#     return out


# def load(path):
#     with open(path, encoding="utf-8") as f:
#         data = json.load(f)
#     if isinstance(data, dict):  # e.g. {"results": [...]}
#         for v in data.values():
#             if isinstance(v, list):
#                 return v
#     return data


# # ----------------------------------------------------------------------------
# # Statistics helpers
# # ----------------------------------------------------------------------------
# def wilson_ci(c, n, z=1.96):
#     if n == 0:
#         return (float("nan"), float("nan"))
#     p = c / n
#     d = 1 + z * z / n
#     centre = (p + z * z / (2 * n)) / d
#     half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
#     return centre - half, centre + half


# def mcnemar_exact(b, c):
#     """Exact two-sided McNemar test.
#     b: only baseline correct, c: only adaptive correct."""
#     n = b + c
#     if n == 0:
#         return 1.0
#     k = min(b, c)
#     p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
#     return min(1.0, 2 * p)


# def mean_std(xs):
#     xs = list(xs)
#     if len(xs) < 2:
#         return (xs[0] if xs else float("nan")), 0.0
#     return st.mean(xs), st.stdev(xs)


# # ----------------------------------------------------------------------------
# # Metrics for one set of records
# # ----------------------------------------------------------------------------
# def compute(records, baseline_k):
#     n = len(records)
#     cb = sum(bool(r["correct_baseline"]) for r in records)
#     ca = sum(bool(r["correct_adaptive"]) for r in records)

#     only_b = sum(bool(r["correct_baseline"]) and not r["correct_adaptive"] for r in records)
#     only_a = sum(bool(r["correct_adaptive"]) and not r["correct_baseline"] for r in records)

#     ks = [int(r["selected_k"]) for r in records]
#     mean_k_adaptive = sum(ks) / n
#     reduction = 100 * (1 - mean_k_adaptive / baseline_k)

#     k_dist = Counter(ks)
#     stop_dist = Counter(r["stop_reason"] for r in records)

#     return {
#         "n": n,
#         "acc_baseline": cb / n,
#         "acc_adaptive": ca / n,
#         "correct_baseline": cb,
#         "correct_adaptive": ca,
#         "ci_baseline": wilson_ci(cb, n),
#         "ci_adaptive": wilson_ci(ca, n),
#         "only_baseline_correct": only_b,
#         "only_adaptive_correct": only_a,
#         "mcnemar_p": mcnemar_exact(only_b, only_a),
#         "mean_docs_baseline": float(baseline_k),
#         "mean_docs_adaptive": mean_k_adaptive,
#         "median_docs_adaptive": st.median(ks),
#         "doc_reduction_pct": reduction,
#         "k_dist": dict(sorted(k_dist.items())),
#         "stop_dist": dict(stop_dist.most_common()),
#     }


# # ----------------------------------------------------------------------------
# # Reporting
# # ----------------------------------------------------------------------------
# def pct(x):
#     return f"{100 * x:.2f}"


# def print_block(title, m):
#     print(f"\n{'=' * 70}\n{title}  (N = {m['n']})\n{'=' * 70}")
#     print("1) Answer accuracy")
#     print(f"   Fixed-k (baseline): {pct(m['acc_baseline'])}%  "
#           f"({m['correct_baseline']}/{m['n']})  "
#           f"95% CI=[{pct(m['ci_baseline'][0])}, {pct(m['ci_baseline'][1])}]")
#     print(f"   Adaptive-k        : {pct(m['acc_adaptive'])}%  "
#           f"({m['correct_adaptive']}/{m['n']})  "
#           f"95% CI=[{pct(m['ci_adaptive'][0])}, {pct(m['ci_adaptive'][1])}]")
#     print(f"   Only baseline correct: {m['only_baseline_correct']} | "
#           f"Only adaptive correct: {m['only_adaptive_correct']} | "
#           f"McNemar p = {m['mcnemar_p']:.4f}")

#     print("2) Average number of documents used")
#     print(f"   Fixed-k  : {m['mean_docs_baseline']:.2f}")
#     print(f"   Adaptive : {m['mean_docs_adaptive']:.2f}  "
#           f"(median = {m['median_docs_adaptive']})  "
#           f"reduction = {m['doc_reduction_pct']:.2f}%")

#     print("3) Distribution of selected k (Adaptive-k)")
#     for k, c in m["k_dist"].items():
#         print(f"   k={k:<3} : {c:>4}  ({100 * c / m['n']:.1f}%)")

#     print("4) Distribution of stop reasons (Adaptive-k)")
#     for r, c in m["stop_dist"].items():
#         print(f"   {r:<20}: {c:>4}  ({100 * c / m['n']:.1f}%)")


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("inputs", nargs="+", help="files, glob patterns, or a directory")
#     ap.add_argument("--baseline-k", type=int, default=None,
#                     help="number of documents in Fixed-k "
#                          "(default: length of retrieved_ids = maximum k)")
#     ap.add_argument("--out", default="metrics_out", help="output directory")
#     args = ap.parse_args()

#     paths = expand_inputs(args.inputs)
#     if not paths:
#         raise SystemExit("No input files found.")
#     os.makedirs(args.out, exist_ok=True)

#     per_file, all_records = {}, []
#     for p in paths:
#         recs = load(p)
#         per_file[os.path.basename(p)] = recs
#         all_records += recs
#     print(f"Loaded {len(paths)} file(s), {len(all_records)} questions in total.")

#     if args.baseline_k:
#         bk = args.baseline_k
#     else:
#         bk = max(len(r["retrieved_ids"]) for r in all_records)
#     print(f"Fixed-k (baseline) number of documents = {bk}")

#     results = {name: compute(recs, bk) for name, recs in per_file.items()}
#     pooled = compute(all_records, bk)

#     for name, m in results.items():
#         print_block(f"File: {name}", m)
#     print_block("Pooled over all files", pooled)

#     # Mean +/- std across files (across runs)
#     ms = {}
#     if len(results) > 1:
#         for key in ["acc_baseline", "acc_adaptive", "mean_docs_adaptive", "doc_reduction_pct"]:
#             ms[key] = mean_std([m[key] for m in results.values()])
#         print(f"\n{'=' * 70}\nMean +/- std across {len(results)} files\n{'=' * 70}")
#         print(f"Accuracy Fixed-k      : {100 * ms['acc_baseline'][0]:.2f} +/- {100 * ms['acc_baseline'][1]:.2f}")
#         print(f"Accuracy Adaptive-k   : {100 * ms['acc_adaptive'][0]:.2f} +/- {100 * ms['acc_adaptive'][1]:.2f}")
#         print(f"Avg docs Adaptive-k   : {ms['mean_docs_adaptive'][0]:.2f} +/- {ms['mean_docs_adaptive'][1]:.2f}")
#         print(f"Doc reduction (%)     : {ms['doc_reduction_pct'][0]:.2f} +/- {ms['doc_reduction_pct'][1]:.2f}")

#     # ---- CSV: summary
#     with open(os.path.join(args.out, "summary.csv"), "w", newline="", encoding="utf-8") as f:
#         w = csv.writer(f)
#         w.writerow(["file", "N", "acc_fixed_k(%)", "acc_adaptive_k(%)",
#                     "avg_docs_fixed_k", "avg_docs_adaptive_k", "doc_reduction(%)", "mcnemar_p"])
#         for name, m in results.items():
#             w.writerow([name, m["n"], pct(m["acc_baseline"]), pct(m["acc_adaptive"]),
#                         f"{m['mean_docs_baseline']:.2f}", f"{m['mean_docs_adaptive']:.2f}",
#                         f"{m['doc_reduction_pct']:.2f}", f"{m['mcnemar_p']:.4f}"])
#         if ms:
#             w.writerow(["MEAN+/-STD", "",
#                         f"{100 * ms['acc_baseline'][0]:.2f}+/-{100 * ms['acc_baseline'][1]:.2f}",
#                         f"{100 * ms['acc_adaptive'][0]:.2f}+/-{100 * ms['acc_adaptive'][1]:.2f}",
#                         f"{bk:.2f}",
#                         f"{ms['mean_docs_adaptive'][0]:.2f}+/-{ms['mean_docs_adaptive'][1]:.2f}",
#                         f"{ms['doc_reduction_pct'][0]:.2f}+/-{ms['doc_reduction_pct'][1]:.2f}", ""])
#         w.writerow(["POOLED", pooled["n"], pct(pooled["acc_baseline"]), pct(pooled["acc_adaptive"]),
#                     f"{bk:.2f}", f"{pooled['mean_docs_adaptive']:.2f}",
#                     f"{pooled['doc_reduction_pct']:.2f}", f"{pooled['mcnemar_p']:.4f}"])

#     # ---- CSV: k distribution
#     all_ks = sorted({k for m in list(results.values()) + [pooled] for k in m["k_dist"]})
#     with open(os.path.join(args.out, "k_distribution.csv"), "w", newline="", encoding="utf-8") as f:
#         w = csv.writer(f)
#         w.writerow(["file"] + [f"k={k} (count)" for k in all_ks] + [f"k={k} (%)" for k in all_ks])
#         for name, m in list(results.items()) + [("POOLED", pooled)]:
#             w.writerow([name] + [m["k_dist"].get(k, 0) for k in all_ks] +
#                        [f"{100 * m['k_dist'].get(k, 0) / m['n']:.1f}" for k in all_ks])

#     # ---- CSV: stop reasons
#     all_rs = sorted({r for m in list(results.values()) + [pooled] for r in m["stop_dist"]})
#     with open(os.path.join(args.out, "stop_reasons.csv"), "w", newline="", encoding="utf-8") as f:
#         w = csv.writer(f)
#         w.writerow(["file"] + [f"{r} (count)" for r in all_rs] + [f"{r} (%)" for r in all_rs])
#         for name, m in list(results.items()) + [("POOLED", pooled)]:
#             w.writerow([name] + [m["stop_dist"].get(r, 0) for r in all_rs] +
#                        [f"{100 * m['stop_dist'].get(r, 0) / m['n']:.1f}" for r in all_rs])

#     # ---- JSON
#     with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
#         json.dump({"baseline_k": bk, "per_file": results, "pooled": pooled,
#                    "mean_std_across_files": {k: list(v) for k, v in ms.items()}},
#                   f, indent=2)

#     print(f"\nOutputs saved to '{args.out}'.")


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""
Compute evaluation metrics for Adaptive-k vs. Fixed-k (baseline)
from eval_round_*.json result files.

Usage:
    python compute_metrics.py "eval_round_test*.json"
    python compute_metrics.py f1.json f2.json ... f10.json
    python compute_metrics.py ./results_dir
    python compute_metrics.py "eval_round_*.json" --baseline-k 32 --out report_out
    python compute_metrics.py "eval_round_*.json" --margins 1 2 3 --alpha 0.05

Outputs (written to the --out directory):
    summary.csv          : per-file metrics + mean/std across files + pooled
    k_distribution.csv   : distribution of selected k (count and %)
    stop_reasons.csv     : distribution of stop reasons (count and %)
    noninferiority.csv   : non-inferiority test results (per file, per margin)
    metrics.json         : all numbers in JSON format
"""
import argparse
import csv
import glob
import json
import math
import os
import statistics as st
from collections import Counter
from statistics import NormalDist

_NORM = NormalDist()


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def expand_inputs(items):
    paths = []
    for it in items:
        if os.path.isdir(it):
            paths += sorted(glob.glob(os.path.join(it, "*.json")))
        else:
            hits = sorted(glob.glob(it))
            paths += hits if hits else [it]
    seen, out = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def load(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):  # e.g. {"results": [...]}
        for v in data.values():
            if isinstance(v, list):
                return v
    return data


# ----------------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------------
def wilson_ci(c, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = c / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def mcnemar_exact(b, c):
    """Exact two-sided McNemar test.
    b: only baseline correct, c: only adaptive correct."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def mean_std(xs):
    xs = list(xs)
    if len(xs) < 2:
        return (xs[0] if xs else float("nan")), 0.0
    return st.mean(xs), st.stdev(xs)


# ----------------------------------------------------------------------------
# Non-inferiority for paired binary outcomes
#
# Notation: delta = acc_adaptive - acc_baseline = (c - b) / n
#   b = only baseline correct, c = only adaptive correct, n = all questions.
# H0: delta <= -margin  (Adaptive-k is worse by at least `margin`)
# H1: delta >  -margin  (Adaptive-k is non-inferior)
# Non-inferiority is claimed when the lower bound of the two-sided
# (1 - alpha) CI of delta is above -margin (one-sided level alpha / 2).
# ----------------------------------------------------------------------------
def _tango_z(b, c, n, d0):
    """Tango (1998) score statistic for H0: delta = d0 (paired proportions).
    The nuisance parameter q = P(only baseline correct) is estimated by
    constrained maximum likelihood under H0."""
    rest = n - b - c
    lo, hi = max(0.0, -d0), (1.0 - d0) / 2.0
    if lo >= hi:
        return float("nan")
    tiny = 1e-300

    def ll(q):
        v = 0.0
        if b:
            v += b * math.log(max(q, tiny))
        if c:
            v += c * math.log(max(q + d0, tiny))
        if rest:
            v += rest * math.log(max(1.0 - 2.0 * q - d0, tiny))
        return v

    a, z = lo, hi  # ll is concave in q -> ternary search
    for _ in range(200):
        m1, m2 = a + (z - a) / 3.0, z - (z - a) / 3.0
        if ll(m1) < ll(m2):
            a = m1
        else:
            z = m2
    q = (a + z) / 2.0
    var = 2.0 * q + d0 - d0 * d0
    if var <= 0:
        return float("nan")
    return (c - b - n * d0) / math.sqrt(n * var)


def _tango_bound(b, c, n, zcrit, lower=True):
    lo, hi = -0.999, 0.999
    for _ in range(100):
        mid = (lo + hi) / 2.0
        zt = _tango_z(b, c, n, mid)
        if math.isnan(zt):
            zt = float("inf") if mid < 0 else float("-inf")
        go_up = (zt > zcrit) if lower else (zt > -zcrit)
        if go_up:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def noninferiority(b, c, n, margins_pp, alpha=0.05):
    """Paired non-inferiority test. Margins are in percentage points.
    Returns values in percentage points."""
    zcrit = _NORM.inv_cdf(1 - alpha / 2)
    delta = (c - b) / n
    se = math.sqrt(max((b + c) - (b - c) ** 2 / n, 0.0)) / n
    wald = (delta - zcrit * se, delta + zcrit * se)
    tango = (_tango_bound(b, c, n, zcrit, True), _tango_bound(b, c, n, zcrit, False))

    tests = []
    for mpp in margins_pp:
        m = mpp / 100.0
        z_w = (delta + m) / se if se > 0 else (float("inf") if delta > -m else float("-inf"))
        z_t = _tango_z(b, c, n, -m)
        tests.append({
            "margin_pp": mpp,
            "p_wald": 1 - _NORM.cdf(z_w),
            "p_tango": 1 - _NORM.cdf(z_t) if not math.isnan(z_t) else float("nan"),
            "noninferior_wald": bool(wald[0] > -m),
            "noninferior_tango": bool(tango[0] > -m),
        })
    return {
        "diff_pp": 100 * delta,
        "ci_wald_pp": (100 * wald[0], 100 * wald[1]),
        "ci_tango_pp": (100 * tango[0], 100 * tango[1]),
        "alpha": alpha,
        "tests": tests,
    }


# ----------------------------------------------------------------------------
# Metrics for one set of records
# ----------------------------------------------------------------------------
def compute(records, baseline_k, margins, alpha):
    n = len(records)
    cb = sum(bool(r["correct_baseline"]) for r in records)
    ca = sum(bool(r["correct_adaptive"]) for r in records)

    only_b = sum(bool(r["correct_baseline"]) and not r["correct_adaptive"] for r in records)
    only_a = sum(bool(r["correct_adaptive"]) and not r["correct_baseline"] for r in records)

    ks = [int(r["selected_k"]) for r in records]
    mean_k_adaptive = sum(ks) / n
    reduction = 100 * (1 - mean_k_adaptive / baseline_k)

    k_dist = Counter(ks)
    stop_dist = Counter(r["stop_reason"] for r in records)

    return {
        "n": n,
        "acc_baseline": cb / n,
        "acc_adaptive": ca / n,
        "correct_baseline": cb,
        "correct_adaptive": ca,
        "ci_baseline": wilson_ci(cb, n),
        "ci_adaptive": wilson_ci(ca, n),
        "only_baseline_correct": only_b,
        "only_adaptive_correct": only_a,
        "mcnemar_p": mcnemar_exact(only_b, only_a),
        "noninferiority": noninferiority(only_b, only_a, n, margins, alpha),
        "mean_docs_baseline": float(baseline_k),
        "mean_docs_adaptive": mean_k_adaptive,
        "median_docs_adaptive": st.median(ks),
        "doc_reduction_pct": reduction,
        "k_dist": dict(sorted(k_dist.items())),
        "stop_dist": dict(stop_dist.most_common()),
    }


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def pct(x):
    return f"{100 * x:.2f}"


def print_block(title, m):
    print(f"\n{'=' * 70}\n{title}  (N = {m['n']})\n{'=' * 70}")
    print("1) Answer accuracy")
    print(f"   Fixed-k (baseline): {pct(m['acc_baseline'])}%  "
          f"({m['correct_baseline']}/{m['n']})  "
          f"95% CI=[{pct(m['ci_baseline'][0])}, {pct(m['ci_baseline'][1])}]")
    print(f"   Adaptive-k        : {pct(m['acc_adaptive'])}%  "
          f"({m['correct_adaptive']}/{m['n']})  "
          f"95% CI=[{pct(m['ci_adaptive'][0])}, {pct(m['ci_adaptive'][1])}]")
    print(f"   Only baseline correct: {m['only_baseline_correct']} | "
          f"Only adaptive correct: {m['only_adaptive_correct']} | "
          f"McNemar p = {m['mcnemar_p']:.4f}")

    print("2) Average number of documents used")
    print(f"   Fixed-k  : {m['mean_docs_baseline']:.2f}")
    print(f"   Adaptive : {m['mean_docs_adaptive']:.2f}  "
          f"(median = {m['median_docs_adaptive']})  "
          f"reduction = {m['doc_reduction_pct']:.2f}%")

    print("3) Distribution of selected k (Adaptive-k)")
    for k, c in m["k_dist"].items():
        print(f"   k={k:<3} : {c:>4}  ({100 * c / m['n']:.1f}%)")

    print("4) Distribution of stop reasons (Adaptive-k)")
    for r, c in m["stop_dist"].items():
        print(f"   {r:<20}: {c:>4}  ({100 * c / m['n']:.1f}%)")

    ni = m["noninferiority"]
    conf = 100 * (1 - ni["alpha"])
    print(f"5) Non-inferiority of Adaptive-k vs Fixed-k "
          f"(one-sided alpha = {ni['alpha'] / 2:.3f})")
    print(f"   Accuracy difference (Adaptive - Fixed): {ni['diff_pp']:+.2f} pp")
    print(f"   {conf:.0f}% CI (Wald)  : [{ni['ci_wald_pp'][0]:+.2f}, {ni['ci_wald_pp'][1]:+.2f}] pp")
    print(f"   {conf:.0f}% CI (Tango) : [{ni['ci_tango_pp'][0]:+.2f}, {ni['ci_tango_pp'][1]:+.2f}] pp")
    for t in ni["tests"]:
        verdict = "NON-INFERIOR" if t["noninferior_tango"] else "not shown"
        print(f"   margin = {t['margin_pp']:.1f} pp : p(Tango) = {t['p_tango']:.4f}, "
              f"p(Wald) = {t['p_wald']:.4f}  ->  {verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="files, glob patterns, or a directory")
    ap.add_argument("--baseline-k", type=int, default=None,
                    help="number of documents in Fixed-k "
                         "(default: length of retrieved_ids = maximum k)")
    ap.add_argument("--margins", type=float, nargs="+", default=[1.0, 2.0, 3.0],
                    help="non-inferiority margins in percentage points "
                         "(choose BEFORE looking at results; default: 1 2 3)")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="two-sided CI level (one-sided test level = alpha/2)")
    ap.add_argument("--out", default="metrics_out", help="output directory")
    args = ap.parse_args()

    paths = expand_inputs(args.inputs)
    if not paths:
        raise SystemExit("No input files found.")
    os.makedirs(args.out, exist_ok=True)

    per_file, all_records = {}, []
    for p in paths:
        recs = load(p)
        per_file[os.path.basename(p)] = recs
        all_records += recs
    print(f"Loaded {len(paths)} file(s), {len(all_records)} questions in total.")

    if args.baseline_k:
        bk = args.baseline_k
    else:
        bk = max(len(r["retrieved_ids"]) for r in all_records)
    print(f"Fixed-k (baseline) number of documents = {bk}")

    results = {name: compute(recs, bk, args.margins, args.alpha) for name, recs in per_file.items()}
    pooled = compute(all_records, bk, args.margins, args.alpha)

    for name, m in results.items():
        print_block(f"File: {name}", m)
    print_block("Pooled over all files", pooled)

    # Mean +/- std across files (across runs)
    ms = {}
    if len(results) > 1:
        for key in ["acc_baseline", "acc_adaptive", "mean_docs_adaptive", "doc_reduction_pct"]:
            ms[key] = mean_std([m[key] for m in results.values()])
        print(f"\n{'=' * 70}\nMean +/- std across {len(results)} files\n{'=' * 70}")
        print(f"Accuracy Fixed-k      : {100 * ms['acc_baseline'][0]:.2f} +/- {100 * ms['acc_baseline'][1]:.2f}")
        print(f"Accuracy Adaptive-k   : {100 * ms['acc_adaptive'][0]:.2f} +/- {100 * ms['acc_adaptive'][1]:.2f}")
        print(f"Avg docs Adaptive-k   : {ms['mean_docs_adaptive'][0]:.2f} +/- {ms['mean_docs_adaptive'][1]:.2f}")
        print(f"Doc reduction (%)     : {ms['doc_reduction_pct'][0]:.2f} +/- {ms['doc_reduction_pct'][1]:.2f}")

    # ---- CSV: summary
    with open(os.path.join(args.out, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "N", "acc_fixed_k(%)", "acc_adaptive_k(%)",
                    "avg_docs_fixed_k", "avg_docs_adaptive_k", "doc_reduction(%)", "mcnemar_p"])
        for name, m in results.items():
            w.writerow([name, m["n"], pct(m["acc_baseline"]), pct(m["acc_adaptive"]),
                        f"{m['mean_docs_baseline']:.2f}", f"{m['mean_docs_adaptive']:.2f}",
                        f"{m['doc_reduction_pct']:.2f}", f"{m['mcnemar_p']:.4f}"])
        if ms:
            w.writerow(["MEAN+/-STD", "",
                        f"{100 * ms['acc_baseline'][0]:.2f}+/-{100 * ms['acc_baseline'][1]:.2f}",
                        f"{100 * ms['acc_adaptive'][0]:.2f}+/-{100 * ms['acc_adaptive'][1]:.2f}",
                        f"{bk:.2f}",
                        f"{ms['mean_docs_adaptive'][0]:.2f}+/-{ms['mean_docs_adaptive'][1]:.2f}",
                        f"{ms['doc_reduction_pct'][0]:.2f}+/-{ms['doc_reduction_pct'][1]:.2f}", ""])
        w.writerow(["POOLED", pooled["n"], pct(pooled["acc_baseline"]), pct(pooled["acc_adaptive"]),
                    f"{bk:.2f}", f"{pooled['mean_docs_adaptive']:.2f}",
                    f"{pooled['doc_reduction_pct']:.2f}", f"{pooled['mcnemar_p']:.4f}"])

    # ---- CSV: non-inferiority (one row per file x margin)
    with open(os.path.join(args.out, "noninferiority.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "N", "only_fixed_correct(b)", "only_adaptive_correct(c)", "diff(pp)",
                    "ci_wald_low(pp)", "ci_wald_high(pp)", "ci_tango_low(pp)", "ci_tango_high(pp)",
                    "margin(pp)", "p_tango", "p_wald", "noninferior_tango", "noninferior_wald"])
        for name, m in list(results.items()) + [("POOLED", pooled)]:
            ni = m["noninferiority"]
            for t in ni["tests"]:
                w.writerow([name, m["n"], m["only_baseline_correct"], m["only_adaptive_correct"],
                            f"{ni['diff_pp']:.2f}",
                            f"{ni['ci_wald_pp'][0]:.2f}", f"{ni['ci_wald_pp'][1]:.2f}",
                            f"{ni['ci_tango_pp'][0]:.2f}", f"{ni['ci_tango_pp'][1]:.2f}",
                            t["margin_pp"], f"{t['p_tango']:.4f}", f"{t['p_wald']:.4f}",
                            t["noninferior_tango"], t["noninferior_wald"]])

    # ---- CSV: k distribution
    all_ks = sorted({k for m in list(results.values()) + [pooled] for k in m["k_dist"]})
    with open(os.path.join(args.out, "k_distribution.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file"] + [f"k={k} (count)" for k in all_ks] + [f"k={k} (%)" for k in all_ks])
        for name, m in list(results.items()) + [("POOLED", pooled)]:
            w.writerow([name] + [m["k_dist"].get(k, 0) for k in all_ks] +
                       [f"{100 * m['k_dist'].get(k, 0) / m['n']:.1f}" for k in all_ks])

    # ---- CSV: stop reasons
    all_rs = sorted({r for m in list(results.values()) + [pooled] for r in m["stop_dist"]})
    with open(os.path.join(args.out, "stop_reasons.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file"] + [f"{r} (count)" for r in all_rs] + [f"{r} (%)" for r in all_rs])
        for name, m in list(results.items()) + [("POOLED", pooled)]:
            w.writerow([name] + [m["stop_dist"].get(r, 0) for r in all_rs] +
                       [f"{100 * m['stop_dist'].get(r, 0) / m['n']:.1f}" for r in all_rs])

    # ---- JSON
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"baseline_k": bk, "margins_pp": args.margins, "alpha": args.alpha,
                   "per_file": results, "pooled": pooled,
                   "mean_std_across_files": {k: list(v) for k, v in ms.items()}},
                  f, indent=2)

    print(f"\nOutputs saved to '{args.out}'.")


if __name__ == "__main__":
    main()
