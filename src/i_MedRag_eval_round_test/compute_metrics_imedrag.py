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
        return json.load(f)


def find_configs(data, baseline_name=None):
    """Return (baseline_config_name, [adaptive_config_names])."""
    cfg = data["config"]
    if baseline_name:
        base = baseline_name
    else:
        bases = [n for n, c in cfg.items() if not c.get("adaptive")]
        if not bases:
            raise SystemExit("No baseline config (adaptive=false) found in file.")
        base = bases[0]
    adaptives = [n for n, c in cfg.items() if c.get("adaptive")]
    if not adaptives:
        raise SystemExit("No adaptive config (adaptive=true) found in file.")
    return base, adaptives


def build_pairs(data, base_name, ada_name, baseline_k, include_errors):
    """Pair baseline and adaptive records by question and return
    (list of pair dicts, number of excluded questions)."""
    base = {r["dataset_idx"]: r for r in data["results"][base_name]}
    ada = {r["dataset_idx"]: r for r in data["results"][ada_name]}
    pairs, excluded = [], 0
    for idx in sorted(set(base) & set(ada)):
        b, a = base[idx], ada[idx]
        has_error = bool(b.get("error")) or bool(a.get("error"))
        if has_error and not include_errors:
            excluded += 1
            continue
        n_calls = len(a["selected_ks"])
        pairs.append({
            "idx": idx,
            "correct_baseline": bool(b["correct"]),
            "correct_adaptive": bool(a["correct"]),
            "selected_ks": list(a["selected_ks"]),
            "stop_reasons": list(a["stop_reasons"]),
            "n_calls": n_calls,
            "docs_adaptive": sum(a["selected_ks"]),
            "docs_baseline": baseline_k * n_calls,
            "judge_calls": a.get("judge_calls", 0),
        })
    return pairs, excluded


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
# Non-inferiority for paired binary outcomes (question level)
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
# Metrics for one set of paired records
# ----------------------------------------------------------------------------
def compute(pairs, baseline_k, margins, alpha):
    n = len(pairs)
    if n == 0:
        return None
    cb = sum(p["correct_baseline"] for p in pairs)
    ca = sum(p["correct_adaptive"] for p in pairs)
    only_b = sum(p["correct_baseline"] and not p["correct_adaptive"] for p in pairs)
    only_a = sum(p["correct_adaptive"] and not p["correct_baseline"] for p in pairs)

    all_ks = [k for p in pairs for k in p["selected_ks"]]
    all_reasons = [r for p in pairs for r in p["stop_reasons"]]
    n_calls = len(all_ks)

    docs_a_q = sum(p["docs_adaptive"] for p in pairs) / n
    docs_b_q = sum(p["docs_baseline"] for p in pairs) / n
    docs_a_r = sum(all_ks) / n_calls
    docs_b_r = float(baseline_k)

    return {
        "n": n,
        "n_retrieval_calls": n_calls,
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
        "docs_per_question_baseline": docs_b_q,
        "docs_per_question_adaptive": docs_a_q,
        "docs_per_question_reduction_pct": 100 * (1 - docs_a_q / docs_b_q),
        "docs_per_retrieval_baseline": docs_b_r,
        "docs_per_retrieval_adaptive": docs_a_r,
        "docs_per_retrieval_reduction_pct": 100 * (1 - docs_a_r / docs_b_r),
        "avg_judge_calls_per_question": sum(p["judge_calls"] for p in pairs) / n,
        "k_dist": dict(sorted(Counter(all_ks).items())),
        "stop_dist": dict(Counter(all_reasons).most_common()),
    }


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def pct(x):
    return f"{100 * x:.2f}"


def print_block(title, m, excluded=None):
    ex = f", excluded due to errors = {excluded}" if excluded is not None else ""
    print(f"\n{'=' * 74}\n{title}  (N = {m['n']} questions, "
          f"{m['n_retrieval_calls']} retrieval calls{ex})\n{'=' * 74}")
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
    print(f"   Per question  : Fixed-k = {m['docs_per_question_baseline']:.2f} | "
          f"Adaptive-k = {m['docs_per_question_adaptive']:.2f} | "
          f"reduction = {m['docs_per_question_reduction_pct']:.2f}%")
    print(f"   Per retrieval : Fixed-k = {m['docs_per_retrieval_baseline']:.2f} | "
          f"Adaptive-k = {m['docs_per_retrieval_adaptive']:.2f} | "
          f"reduction = {m['docs_per_retrieval_reduction_pct']:.2f}%")

    nc = m["n_retrieval_calls"]
    print("3) Distribution of selected k (Adaptive-k, over retrieval calls)")
    for k, c in m["k_dist"].items():
        print(f"   k={k:<3} : {c:>4}  ({100 * c / nc:.1f}%)")

    print("4) Distribution of stop reasons (Adaptive-k, over retrieval calls)")
    for r, c in m["stop_dist"].items():
        print(f"   {r:<32}: {c:>4}  ({100 * c / nc:.1f}%)")

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
    ap.add_argument("--baseline-config", default=None,
                    help="name of the baseline config (default: auto-detect adaptive=false)")
    ap.add_argument("--baseline-k", type=int, default=None,
                    help="k of Fixed-k (default: read from the baseline config)")
    ap.add_argument("--include-errors", action="store_true",
                    help="keep questions with API errors (counted as incorrect)")
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

    datas = {os.path.basename(p): load(p) for p in paths}
    print(f"Loaded {len(datas)} file(s).")

    first = next(iter(datas.values()))
    base_name, ada_names = find_configs(first, args.baseline_config)
    bk = args.baseline_k or first["config"][base_name]["k"]
    print(f"Baseline config: {base_name} (k = {bk})")
    print(f"Adaptive config(s): {', '.join(ada_names)}")

    summary_rows, k_rows, stop_rows, ni_rows = [], [], [], []
    json_out = {"baseline_k": bk, "margins_pp": args.margins, "alpha": args.alpha}

    for ada_name in ada_names:
        per_file, excluded, pooled_pairs = {}, {}, []
        for fname, data in datas.items():
            if ada_name not in data["results"]:
                continue
            pairs, ex = build_pairs(data, base_name, ada_name, bk, args.include_errors)
            per_file[fname], excluded[fname] = pairs, ex
            pooled_pairs += pairs

        results = {f: compute(p, bk, args.margins, args.alpha) for f, p in per_file.items()}
        results = {f: m for f, m in results.items() if m}
        pooled = compute(pooled_pairs, bk, args.margins, args.alpha)

        print(f"\n\n######## Adaptive config: {ada_name} ########")
        for fname, m in results.items():
            print_block(f"File: {fname}", m, excluded[fname])
        print_block("Pooled over all files", pooled, sum(excluded.values()))

        ms = {}
        if len(results) > 1:
            keys = ["acc_baseline", "acc_adaptive",
                    "docs_per_question_adaptive", "docs_per_question_reduction_pct",
                    "docs_per_retrieval_adaptive", "docs_per_retrieval_reduction_pct"]
            for key in keys:
                ms[key] = mean_std([m[key] for m in results.values()])
            print(f"\n{'=' * 74}\nMean +/- std across {len(results)} files\n{'=' * 74}")
            print(f"Accuracy Fixed-k             : {100 * ms['acc_baseline'][0]:.2f} +/- {100 * ms['acc_baseline'][1]:.2f}")
            print(f"Accuracy Adaptive-k          : {100 * ms['acc_adaptive'][0]:.2f} +/- {100 * ms['acc_adaptive'][1]:.2f}")
            print(f"Docs/question Adaptive-k     : {ms['docs_per_question_adaptive'][0]:.2f} +/- {ms['docs_per_question_adaptive'][1]:.2f}")
            print(f"Docs/question reduction (%)  : {ms['docs_per_question_reduction_pct'][0]:.2f} +/- {ms['docs_per_question_reduction_pct'][1]:.2f}")
            print(f"Docs/retrieval Adaptive-k    : {ms['docs_per_retrieval_adaptive'][0]:.2f} +/- {ms['docs_per_retrieval_adaptive'][1]:.2f}")
            print(f"Docs/retrieval reduction (%) : {ms['docs_per_retrieval_reduction_pct'][0]:.2f} +/- {ms['docs_per_retrieval_reduction_pct'][1]:.2f}")

        # ---- rows for CSV
        for fname, m in list(results.items()) + [("POOLED", pooled)]:
            ex = sum(excluded.values()) if fname == "POOLED" else excluded[fname]
            summary_rows.append([
                ada_name, fname, m["n"], ex, m["n_retrieval_calls"],
                pct(m["acc_baseline"]), pct(m["acc_adaptive"]), f"{m['mcnemar_p']:.4f}",
                f"{m['docs_per_question_baseline']:.2f}", f"{m['docs_per_question_adaptive']:.2f}",
                f"{m['docs_per_question_reduction_pct']:.2f}",
                f"{m['docs_per_retrieval_baseline']:.2f}", f"{m['docs_per_retrieval_adaptive']:.2f}",
                f"{m['docs_per_retrieval_reduction_pct']:.2f}",
                f"{m['avg_judge_calls_per_question']:.2f}",
            ])
            ni = m["noninferiority"]
            for t in ni["tests"]:
                ni_rows.append([
                    ada_name, fname, m["n"], m["only_baseline_correct"], m["only_adaptive_correct"],
                    f"{ni['diff_pp']:.2f}",
                    f"{ni['ci_wald_pp'][0]:.2f}", f"{ni['ci_wald_pp'][1]:.2f}",
                    f"{ni['ci_tango_pp'][0]:.2f}", f"{ni['ci_tango_pp'][1]:.2f}",
                    t["margin_pp"], f"{t['p_tango']:.4f}", f"{t['p_wald']:.4f}",
                    t["noninferior_tango"], t["noninferior_wald"],
                ])
        if ms:
            f2 = lambda k, s=1: f"{s * ms[k][0]:.2f}+/-{s * ms[k][1]:.2f}"
            summary_rows.insert(len(summary_rows) - 1, [
                ada_name, "MEAN+/-STD", "", "", "",
                f2("acc_baseline", 100), f2("acc_adaptive", 100), "",
                "", f2("docs_per_question_adaptive"), f2("docs_per_question_reduction_pct"),
                "", f2("docs_per_retrieval_adaptive"), f2("docs_per_retrieval_reduction_pct"), "",
            ])

        ks_all = sorted({k for m in list(results.values()) + [pooled] for k in m["k_dist"]})
        rs_all = sorted({r for m in list(results.values()) + [pooled] for r in m["stop_dist"]})
        if not k_rows:
            k_rows.append(["adaptive_config", "file"] + [f"k={k} (count)" for k in ks_all] +
                          [f"k={k} (%)" for k in ks_all])
        if not stop_rows:
            stop_rows.append(["adaptive_config", "file"] + [f"{r} (count)" for r in rs_all] +
                             [f"{r} (%)" for r in rs_all])
        for fname, m in list(results.items()) + [("POOLED", pooled)]:
            nc = m["n_retrieval_calls"]
            k_rows.append([ada_name, fname] + [m["k_dist"].get(k, 0) for k in ks_all] +
                          [f"{100 * m['k_dist'].get(k, 0) / nc:.1f}" for k in ks_all])
            stop_rows.append([ada_name, fname] + [m["stop_dist"].get(r, 0) for r in rs_all] +
                             [f"{100 * m['stop_dist'].get(r, 0) / nc:.1f}" for r in rs_all])

        json_out[ada_name] = {"per_file": results, "pooled": pooled,
                              "excluded_due_to_errors": excluded,
                              "mean_std_across_files": {k: list(v) for k, v in ms.items()}}

    # ---- write files
    with open(os.path.join(args.out, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["adaptive_config", "file", "N_questions", "N_excluded", "N_retrieval_calls",
                    "acc_fixed_k(%)", "acc_adaptive_k(%)", "mcnemar_p",
                    "docs_per_question_fixed_k", "docs_per_question_adaptive_k", "docs_per_question_reduction(%)",
                    "docs_per_retrieval_fixed_k", "docs_per_retrieval_adaptive_k", "docs_per_retrieval_reduction(%)",
                    "avg_judge_calls_per_question"])
        w.writerows(summary_rows)
    with open(os.path.join(args.out, "noninferiority.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["adaptive_config", "file", "N", "only_fixed_correct(b)", "only_adaptive_correct(c)",
                    "diff(pp)", "ci_wald_low(pp)", "ci_wald_high(pp)", "ci_tango_low(pp)", "ci_tango_high(pp)",
                    "margin(pp)", "p_tango", "p_wald", "noninferior_tango", "noninferior_wald"])
        w.writerows(ni_rows)
    for name, rows in (("k_distribution.csv", k_rows), ("stop_reasons.csv", stop_rows)):
        with open(os.path.join(args.out, name), "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2)

    print(f"\nOutputs saved to '{args.out}'.")


if __name__ == "__main__":
    main()