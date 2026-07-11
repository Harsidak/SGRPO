"""
Experiments/local_benchmark.py

Local benchmark harness: run the five-algorithm validation sequence on the
local GPU, store every run's console log + structured JSONL metrics in a
timestamped results directory, and render offline comparison graphs.

This is the fast, local counterpart to Experiments/run_benchmark.py (the
5-seed statistical benchmark for the paper). Use it after every code change
to (a) confirm nothing regressed across the mandated ppo -> grpo -> dapo ->
bapo -> sgrpo order and (b) get training-dynamics graphs without W&B.

Outputs, under Experiments/results/local_bench/<timestamp>_<label>/:
    <algo>.log            console output of the run
    <algo>_*.jsonl        structured per-step metrics (via wandb_logger's
                          local sink, redirected here by SGRPO_LOCAL_LOG_DIR)
    graphs/<metric>.png   one chart per metric, all algorithms overlaid
    summary.json          machine-readable run summary
    summary.md            per-algorithm results table (the charts' table view)

Usage:
    python -m Experiments.local_benchmark                    # smoke defaults
    python -m Experiments.local_benchmark --steps 20 --inner_epochs 3
    python -m Experiments.local_benchmark --algorithms grpo sgrpo --steps 50
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Chart styling (dataviz reference palette, light mode) ───────────────────
# Fixed entity -> hue mapping: color follows the algorithm, never its rank,
# so a run with a subset of algorithms keeps identical colors.
ALGO_COLORS = {
    "ppo":   "#2a78d6",   # blue
    "grpo":  "#1baf7a",   # aqua
    "dapo":  "#eda100",   # yellow
    "bapo":  "#008300",   # green
    "sgrpo": "#4a3aa7",   # violet
}
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

ALGO_ORDER = ["ppo", "grpo", "dapo", "bapo", "sgrpo"]

# Metrics to graph: (jsonl key, chart title, y-label)
STEP_METRICS = [
    ("train/loss",           "Policy loss",            "loss"),
    ("train/reward_mean",    "Mean rollout reward",    "reward"),
    ("train/entropy",        "Token entropy",          "entropy (nats)"),
    ("train/clip_fraction",  "Clip fraction",          "fraction of tokens"),
    ("train/ratio_std",      "Importance ratio std",   "std(r_t)"),
    ("train/grad_norm",      "Gradient norm",          "||g||"),
    ("train/advantage_std",  "Advantage std",          "std(A)"),
    ("train/degenerate_group_rate", "Degenerate group rate", "rate"),
]


def run_one(algo: str, args, out_dir: str) -> dict:
    """Run one training job as a subprocess, teeing output to <algo>.log."""
    group_size = 1 if algo == "ppo" else args.group_size
    cmd = [
        sys.executable, "main.py",
        "--algorithm", algo,
        "--steps", str(args.steps),
        "--group_size", str(group_size),
        "--max_tokens", str(args.max_tokens),
        "--eval_samples", str(args.eval_samples),
        "--eval_every", str(args.eval_every),
        "--inner_epochs", str(args.inner_epochs),
        "--seed", str(args.seed),
        "--device", args.device,
        "--run_name", f"localbench-{args.label}",
        "--no_wandb",
    ]
    env = dict(os.environ)
    env["SGRPO_LOCAL_LOG_DIR"] = out_dir   # redirect the JSONL sink here

    log_path = os.path.join(out_dir, f"{algo}.log")
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
    return {
        "algorithm": algo,
        "returncode": proc.returncode,
        "wall_seconds": round(time.time() - t0, 1),
        "log": os.path.basename(log_path),
    }


def load_metrics(out_dir: str) -> dict:
    """Parse every JSONL sink file in out_dir into {algo: [records...]}."""
    by_algo = defaultdict(list)
    for fname in sorted(os.listdir(out_dir)):
        if not fname.endswith(".jsonl"):
            continue
        algo = fname.split("_")[0]
        with open(os.path.join(out_dir, fname), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    by_algo[algo].append(json.loads(line))
    return by_algo


def series_for(records: list, key: str):
    """Extract (x, y) for one metric; x is the optimizer-update index so
    inner-epoch runs plot every update, not one point per outer step."""
    xs, ys = [], []
    for rec in records:
        if rec["kind"] not in ("step", "degenerate"):
            continue
        if key in rec and isinstance(rec[key], (int, float)):
            xs.append(rec.get("sys/global_update",
                              rec.get("sys/step", len(xs))))
            ys.append(rec[key])
    return xs, ys


def style_axes(ax, title: str, ylabel: str):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK_PRIMARY, fontsize=12, loc="left", pad=10)
    ax.set_xlabel("optimizer update", color=INK_MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)
    ax.grid(True, axis="y", color=GRIDLINE, linewidth=0.75)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(0.75)


def plot_metric(by_algo: dict, key: str, title: str, ylabel: str,
                out_path: str) -> bool:
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ends = []   # (algo, x_end, y_end) for staggered direct labels
    for algo in ALGO_ORDER:
        if algo not in by_algo:
            continue
        xs, ys = series_for(by_algo[algo], key)
        if not ys:
            continue
        color = ALGO_COLORS[algo]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o",
                markersize=4 if len(ys) <= 40 else 0, label=algo.upper())
        ends.append((algo, xs[-1], ys[-1]))
    if not ends:
        plt.close(fig)
        return False
    # Direct labels at line ends (relief for low-contrast hues); text stays
    # in secondary ink, the line carries the color. Coincident line ends
    # (e.g. identical flat series) get vertically staggered labels so they
    # never overprint.
    y_min, y_max = ax.get_ylim()
    min_gap = (y_max - y_min) * 0.05
    placed = []
    for algo, x_end, y_end in sorted(ends, key=lambda e: e[2]):
        y_label = y_end
        while any(abs(y_label - p) < min_gap for p in placed):
            y_label += min_gap
        placed.append(y_label)
        ax.annotate(f" {algo.upper()}", (x_end, y_label),
                    color=INK_SECONDARY, fontsize=8, va="center")
    style_axes(ax, title, ylabel)
    leg = ax.legend(loc="best", fontsize=8, frameon=False)
    for text in leg.get_texts():
        text.set_color(INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    return True


def last_of(records: list, kind: str, key: str):
    val = None
    for rec in records:
        if rec["kind"] == kind and key in rec:
            val = rec[key]
    return val


def write_summary(by_algo: dict, run_results: list, args, out_dir: str,
                  graphs: list) -> None:
    rows = []
    for algo in ALGO_ORDER:
        recs = by_algo.get(algo, [])
        steps = [r for r in recs if r["kind"] == "step"]
        rows.append({
            "algorithm": algo,
            "returncode": next((r["returncode"] for r in run_results
                                if r["algorithm"] == algo), None),
            "wall_seconds": next((r["wall_seconds"] for r in run_results
                                  if r["algorithm"] == algo), None),
            "optimizer_updates": len(steps),
            "degenerate_events": sum(r["kind"] == "degenerate" for r in recs),
            "final_loss": last_of(recs, "step", "train/loss"),
            "final_entropy": last_of(recs, "step", "train/entropy"),
            "final_clip_fraction": last_of(recs, "step", "train/clip_fraction"),
            "eval_accuracy": last_of(recs, "eval", "eval/gsm8k_accuracy"),
            "arch_branch": last_of(recs, "metadata", "sys/architecture_branch"),
        })

    summary = {
        "label": args.label,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "steps": args.steps, "group_size": args.group_size,
            "max_tokens": args.max_tokens, "inner_epochs": args.inner_epochs,
            "seed": args.seed, "eval_samples": args.eval_samples,
        },
        "runs": rows,
        "graphs": graphs,
    }
    with open(os.path.join(out_dir, "summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    def fmt(v, spec=".4f"):
        if v is None:
            return "—"
        return format(v, spec) if isinstance(v, float) else str(v)

    lines = [
        f"# Local benchmark — {args.label}",
        "",
        f"Created {summary['created']} | steps={args.steps} "
        f"G={args.group_size} max_tokens={args.max_tokens} "
        f"mu={args.inner_epochs} seed={args.seed}",
        "",
        "| Algorithm | Exit | Wall (s) | Updates | Degenerate | "
        "Final loss | Entropy | ClipFrac | Eval acc | Branch |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['algorithm'].upper()} | {fmt(r['returncode'], 'd')} "
            f"| {fmt(r['wall_seconds'], '.1f')} | {r['optimizer_updates']} "
            f"| {r['degenerate_events']} | {fmt(r['final_loss'])} "
            f"| {fmt(r['final_entropy'], '.3f')} "
            f"| {fmt(r['final_clip_fraction'], '.3f')} "
            f"| {fmt(r['eval_accuracy'], '.3f')} "
            f"| {r['arch_branch'] or '—'} |"
        )
    lines += ["", "Graphs: " + (", ".join(graphs) if graphs else "none"), ""]
    with open(os.path.join(out_dir, "summary.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Local benchmark: run + log + graph all algorithms")
    parser.add_argument("--algorithms", nargs="+", default=ALGO_ORDER,
                        choices=ALGO_ORDER)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=16)
    parser.add_argument("--inner_epochs", type=int, default=3)
    parser.add_argument("--eval_samples", type=int, default=2)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--label", type=str, default="smoke")
    parser.add_argument("--replot", type=str, default=None, metavar="DIR",
                        help="Regenerate graphs + summary from an existing "
                             "benchmark directory instead of running "
                             "training.")
    args = parser.parse_args()

    if args.replot:
        out_dir = args.replot
        run_results = []
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("Experiments", "results", "local_bench",
                               f"{stamp}_{args.label}")
        os.makedirs(out_dir, exist_ok=True)
        print(f"Local benchmark -> {out_dir}")

        # Mandated validation order: ppo -> grpo -> dapo -> bapo -> sgrpo.
        ordered = [a for a in ALGO_ORDER if a in args.algorithms]
        run_results = []
        for algo in ordered:
            print(f"[{algo}] running {args.steps} steps "
                  f"(mu={args.inner_epochs})...", flush=True)
            res = run_one(algo, args, out_dir)
            status = ("ok" if res["returncode"] == 0
                      else f"FAILED ({res['returncode']})")
            print(f"[{algo}] {status} in {res['wall_seconds']}s")
            run_results.append(res)

    by_algo = load_metrics(out_dir)
    graphs_dir = os.path.join(out_dir, "graphs")
    os.makedirs(graphs_dir, exist_ok=True)
    graphs = []
    for key, title, ylabel in STEP_METRICS:
        fname = key.split("/")[-1] + ".png"
        if plot_metric(by_algo, key, title, ylabel,
                       os.path.join(graphs_dir, fname)):
            graphs.append(f"graphs/{fname}")

    write_summary(by_algo, run_results, args, out_dir, graphs)
    print(f"\nSummary: {os.path.join(out_dir, 'summary.md')}")
    print(f"Graphs:  {len(graphs)} rendered in {graphs_dir}")

    failed = [r for r in run_results if r["returncode"] != 0]
    if failed:
        print(f"FAILURES: {[r['algorithm'] for r in failed]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
