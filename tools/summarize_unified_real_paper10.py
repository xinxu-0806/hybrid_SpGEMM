#!/usr/bin/env python3
"""Summarize production real-HBM three-mode runs against FPGA baselines."""

from __future__ import annotations

import argparse
import csv
import pathlib
import statistics


MATRICES = (
    "poisson3Da", "raefsky1", "crystk01", "s3rmt3m3", "t2dah_a",
    "nasa2910", "bcsstk24", "cavity26", "ex9", "af23560",
)

BASELINES = {
    "poisson3Da": (60.97, 115.01, 62.17, 105.96, 57.28, 70.44),
    "raefsky1": (77.98, 102.00, 55.14, 74.52, 40.28, 76.96),
    "crystk01": (22.35, 78.49, 42.43, 53.79, 29.07, 22.59),
    "s3rmt3m3": (10.37, 35.18, 19.02, 21.73, 11.75, 9.34),
    "t2dah_a": (8.44, 20.10, 10.87, 14.66, 7.92, 5.89),
    "nasa2910": (12.31, 54.01, 29.20, 36.18, 19.56, 13.42),
    "bcsstk24": (8.21, 29.85, 16.13, 18.10, 9.78, 7.83),
    "cavity26": (20.12, 26.93, 14.56, 17.95, 9.70, 22.91),
    "ex9": (5.01, 14.11, 7.63, 9.07, 4.90, 4.82),
    "af23560": (61.03, 50.69, 27.40, 38.29, 20.70, 59.59),
}

MODE_ORDER = (
    "row_adaptive", "merge_preferred_capacity_safe", "dense_only",
)


def read_totals(path: pathlib.Path, log: pathlib.Path, reps: int) -> dict[str, dict[str, str]]:
    if "UNIFIED_REAL_HOST_BOARD: PASS" not in log.read_text(errors="replace"):
        raise ValueError(f"missing board PASS marker in {log}")
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    totals = {row["mode"]: row for row in rows if row["batch"] == "TOTAL"}
    if tuple(mode for mode in MODE_ORDER if mode in totals) != MODE_ORDER:
        raise ValueError(f"missing three TOTAL modes in {path}: {sorted(totals)}")
    for mode in MODE_ORDER:
        samples = [row for row in rows
                   if row["batch"] == "SAMPLE" and row["mode"] == mode]
        if len(samples) != reps:
            raise ValueError(f"{path}: {mode} has {len(samples)} samples, expected {reps}")
    output_counts = {int(totals[mode]["output_nnz"]) for mode in MODE_ORDER}
    if len(output_counts) != 1:
        raise ValueError(f"three modes disagree on output nnz in {path}")
    return totals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=pathlib.Path)
    parser.add_argument("--reps", type=int, default=10)
    args = parser.parse_args()

    summary: list[dict[str, object]] = []
    for matrix in MATRICES:
        stem = f"{matrix}_r{args.reps}"
        totals = read_totals(args.result_dir / f"{stem}.csv",
                             args.result_dir / f"{stem}.log", args.reps)
        times = {mode: float(totals[mode]["kernel_ms"]) for mode in MODE_ORDER}
        selected_mode = "row_adaptive"
        oracle_mode = min(MODE_ORDER, key=times.__getitem__)
        selected_ms = times[selected_mode]
        oracle_ms = times[oracle_mode]
        best_external = min(BASELINES[matrix])
        adaptive = totals[selected_mode]
        summary.append({
            "matrix": matrix,
            "adaptive_ms": selected_ms,
            "merge_ms": times["merge_preferred_capacity_safe"],
            "dense_ms": times["dense_only"],
            "oracle_mode": oracle_mode,
            "oracle_ms": oracle_ms,
            "host_regret_pct": 100.0 * (selected_ms - oracle_ms) / oracle_ms,
            "adaptive_merge_rows": int(adaptive["merge_logical_rows"]),
            "adaptive_dense_rows": int(adaptive["dense_logical_rows"]),
            "output_nnz": int(adaptive["output_nnz"]),
            "best_external_ms": best_external,
            "adaptive_speedup_vs_best_external": best_external / selected_ms,
            "oracle_speedup_vs_best_external": best_external / oracle_ms,
        })

    output_csv = args.result_dir / "unified_real_kernel_time_comparison.csv"
    with output_csv.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    adaptive_mean = statistics.mean(float(row["adaptive_ms"]) for row in summary)
    merge_mean = statistics.mean(float(row["merge_ms"]) for row in summary)
    dense_mean = statistics.mean(float(row["dense_ms"]) for row in summary)
    oracle_mean = statistics.mean(float(row["oracle_ms"]) for row in summary)
    best_external_mean = statistics.mean(min(BASELINES[matrix]) for matrix in MATRICES)

    output_md = args.result_dir / "SUMMARY.md"
    with output_md.open("w") as output:
        output.write("# Unified real-HBM U280 kernel time\n\n")
        output.write(
            f"FP32 A×A；每个模式 1 次 warm-up 后 {args.reps} 次；表中为 "
            "Host 从 kernel launch 到 wait 返回的总时间中位数；多次容量批次按同一 "
            "rep 求和。Host 分析、显式 HBM 传输和 CPU 验证不计入 kernel time。\n\n"
        )
        output.write("| 矩阵 | Adaptive ms | MERGE-only* ms | DENSE-only ms | Oracle | Oracle ms | M/D | Host regret | 最佳外部 ms | Adaptive/外部 |\n")
        output.write("|---|---:|---:|---:|---|---:|---:|---:|---:|---:|\n")
        for row in summary:
            output.write(
                f"| {row['matrix']} | {float(row['adaptive_ms']):.3f} | "
                f"{float(row['merge_ms']):.3f} | {float(row['dense_ms']):.3f} | "
                f"{row['oracle_mode']} | {float(row['oracle_ms']):.3f} | "
                f"{row['adaptive_merge_rows']}/{row['adaptive_dense_rows']} | "
                f"{float(row['host_regret_pct']):.2f}% | "
                f"{float(row['best_external_ms']):.2f} | "
                f"{float(row['adaptive_speedup_vs_best_external']):.3f}x |\n"
            )
        output.write(
            f"| **算术平均** | **{adaptive_mean:.3f}** | **{merge_mean:.3f}** | "
            f"**{dense_mean:.3f}** |  | **{oracle_mean:.3f}** |  |  | "
            f"**{best_external_mean:.3f}** | **{best_external_mean/adaptive_mean:.3f}x** |\n\n"
        )
        output.write("\\* MERGE-only 对硬件不合法或容量超限的行保留强制 DENSE 兜底。\n")

    print(
        f"UNIFIED_REAL_SUMMARY adaptive_mean_ms={adaptive_mean:.6f} "
        f"merge_mean_ms={merge_mean:.6f} dense_mean_ms={dense_mean:.6f} "
        f"oracle_mean_ms={oracle_mean:.6f} "
        f"best_external_mean_ms={best_external_mean:.6f} "
        f"adaptive_beats_external={int(adaptive_mean < best_external_mean)}"
    )


if __name__ == "__main__":
    main()
