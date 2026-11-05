"""Compare the last run in two append-only trainer logs, aligned by optimizer step."""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean


def read_last_run(path: Path) -> tuple[dict[int, dict], list[int]]:
    runs: list[list[dict]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        step = int(row["step"])
        if not math.isfinite(float(row["loss"])):
            raise ValueError(f"Non-finite loss at step {step}: {path}")
        if not runs or step <= int(runs[-1][-1]["step"]):
            runs.append([])
        runs[-1].append(row)
    if not runs:
        raise ValueError(f"Empty log: {path}")
    return {int(row["step"]): row for row in runs[-1]}, [len(run) for run in runs]


def compare(reference: Path, candidate: Path, expected_steps: int, threshold: float) -> tuple[dict, list[dict]]:
    ref, ref_runs = read_last_run(reference)
    cand, cand_runs = read_last_run(candidate)
    expected = set(range(1, expected_steps + 1))
    if set(ref) != expected or set(cand) != expected:
        raise ValueError(f"Both last runs must contain exactly steps 1..{expected_steps}; got {len(ref)}, {len(cand)}")
    rows = []
    for step in sorted(expected):
        a, b = float(ref[step]["loss"]), float(cand[step]["loss"])
        if a == 0:
            raise ValueError(f"Zero reference loss at step {step}; relative error is undefined")
        rows.append(
            {
                "step": step,
                "reference_loss": a,
                "candidate_loss": b,
                "absolute_error": abs(b - a),
                "relative_error_percent": abs(b - a) / abs(a) * 100,
            }
        )
    mape = mean(row["relative_error_percent"] for row in rows)
    reference_mean = mean(row["reference_loss"] for row in rows)
    candidate_mean = mean(row["candidate_loss"] for row in rows)
    summary = {
        "reference": str(reference.resolve()),
        "candidate": str(candidate.resolve()),
        "reference_run_lengths": ref_runs,
        "candidate_run_lengths": cand_runs,
        "steps": expected_steps,
        "metric": "mean(abs(candidate_loss-reference_loss)/abs(reference_loss))*100",
        "mean_relative_error_percent": mape,
        "mean_absolute_error": mean(row["absolute_error"] for row in rows),
        "max_relative_error_percent": max(row["relative_error_percent"] for row in rows),
        "reference_mean_loss": reference_mean,
        "candidate_mean_loss": candidate_mean,
        "relative_difference_of_means_percent": abs(candidate_mean - reference_mean) / abs(reference_mean) * 100,
        "threshold_percent": threshold,
        "passed": mape < threshold,
    }
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--expected-steps", type=int, default=250)
    parser.add_argument("--threshold-percent", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--reference-label", default="FSDP2")
    parser.add_argument("--candidate-label", default="FSDPTurbo EP")
    args = parser.parse_args()
    summary, rows = compare(args.reference, args.candidate, args.expected_steps, args.threshold_percent)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "per_step.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, layout="constrained")
        steps = [row["step"] for row in rows]
        axes[0].plot(steps, [row["reference_loss"] for row in rows], label=args.reference_label, linewidth=1)
        axes[0].plot(
            steps, [row["candidate_loss"] for row in rows], label=args.candidate_label, linewidth=1, alpha=0.8
        )
        axes[0].set_ylabel("Training loss")
        axes[0].legend()
        axes[0].set_title(
            f"Step-aligned loss comparison; mean relative error {summary['mean_relative_error_percent']:.4f}%"
        )
        axes[1].plot(steps, [row["relative_error_percent"] for row in rows], linewidth=1)
        axes[1].axhline(args.threshold_percent, color="red", linestyle="--", label=f"{args.threshold_percent}% target")
        axes[1].set(xlabel="Optimizer step", ylabel="Absolute relative error (%)")
        axes[1].legend()
        fig.savefig(args.output_dir / "loss_comparison.png", dpi=160)
        plt.close(fig)
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
