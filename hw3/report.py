"""Collect every run in runs/ into a table and a figure.

    python -m hw3.report --versus sft

    runs/summary.md    one row per run, and each run's new-task accuracy minus that of --versus, with a 95% interval
    runs/results.png   (a) new-task accuracy against prior-task average; (b) forgetting against the KL to the base model
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})


def read(run: Path) -> dict:
    r = {"name": run.name} | json.loads((run / "metrics.json").read_text())
    for file in ("args.json", "prior.json"):
        if (run / file).exists():
            r |= json.loads((run / file).read_text())
    r["correct"] = {(d := json.loads(line))["id"]: d["correct"] for line in open(run / "eval.jsonl")}
    return r


def paired_difference(a: dict, b: dict, draws: int = 2000) -> tuple[float, float, float]:
    """Accuracy of b minus accuracy of a on the same examples, with a bootstrap 95% interval."""
    diffs = [b[i] - a[i] for i in sorted(set(a) & set(b))]
    rng = random.Random(0)
    means = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(draws))
    return sum(diffs) / len(diffs), means[int(0.025 * draws)], means[int(0.975 * draws)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--versus", default="sft", help="the run to compare every other run with")
    args = ap.parse_args()
    runs = [read(d) for d in sorted(args.runs.iterdir()) if (d / "metrics.json").exists()]
    base = next((r for r in runs if r["name"] == "base"), {})

    f = lambda v: "" if v is None else f"{v:.3f}"
    lines = ["| run | data | loss | context | new task | ifeval | humaneval | prior avg | forgetting | KL to base |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        if "average" in r and "average" in base:
            r["forgetting"] = base["average"] - r["average"]
        se = (r["accuracy"] * (1 - r["accuracy"]) / r["n"]) ** 0.5
        lines.append(f"| {r['name']} | {r.get('data', '')} | {r.get('loss', '')} | {r.get('context', '')} | "
                     f"{r['accuracy']:.3f} ± {se:.3f} | {f(r.get('ifeval'))} | {f(r.get('humaneval'))} | "
                     f"{f(r.get('average'))} | {f(r.get('forgetting'))} | {r['kl_to_base']:.3f} |")
    ref = next((r for r in runs if r["name"] == args.versus), None)
    if ref:
        lines.append(f"\nNew-task accuracy minus that of {ref['name']}, on the same examples (95% bootstrap interval):\n")
        for r in runs:
            if r is not ref:
                d, lo, hi = paired_difference(ref["correct"], r["correct"])
                lines.append(f"- {r['name']}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    (args.runs / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.6))
    for r in runs:
        if "average" in r:
            a.scatter(r["accuracy"], r["average"], color="black", s=18)
            a.annotate(r["name"], (r["accuracy"], r["average"]), xytext=(4, 4), textcoords="offset points")
        if "forgetting" in r:
            b.scatter(r["kl_to_base"], r["forgetting"], color="black", s=18)
            b.annotate(r["name"], (r["kl_to_base"], r["forgetting"]), xytext=(4, 4), textcoords="offset points")
    a.set(xlabel="new-task accuracy", ylabel="prior-task average", title="(a) learning and forgetting")
    b.set(xlabel="KL to the base model on the new task (nats / token)", ylabel="forgetting", title="(b) distance and forgetting")
    fig.tight_layout(), fig.savefig(args.runs / "results.png", dpi=160), plt.close(fig)


if __name__ == "__main__":
    main()
