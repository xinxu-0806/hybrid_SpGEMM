#!/usr/bin/env python3
"""Validate routed timing, optionally accounting for Vitis clock scaling."""

import argparse
import math
from pathlib import Path
import re
import subprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--min-wns", type=float, default=0.0)
    parser.add_argument("--min-whs", type=float, default=0.0)
    parser.add_argument(
        "--clock",
        help="validate this row in the Intra Clock Table instead of the design summary",
    )
    parser.add_argument(
        "--allow-auto-scale-xclbin",
        type=Path,
        metavar="XCLBIN",
        help=(
            "allow a setup violation for --clock only when the final xclbin "
            "contains a sufficiently reduced scalable-clock frequency"
        ),
    )
    return parser.parse_args()


def parse_design_summary(report: Path) -> tuple[float, float, float, float]:
    lines = report.read_text(errors="replace").splitlines()
    in_summary = False
    saw_separator = False
    for line in lines:
        if "| Design Timing Summary" in line:
            in_summary = True
            saw_separator = False
            continue
        if not in_summary:
            continue
        fields = line.split()
        if fields and all(set(field) <= {"-"} for field in fields):
            saw_separator = True
            continue
        if not saw_separator or len(fields) < 6:
            continue
        try:
            wns = float(fields[0])
            tns = float(fields[1])
            whs = float(fields[4])
            ths = float(fields[5])
        except ValueError:
            continue
        return wns, tns, whs, ths
    raise SystemExit(f"could not parse Design Timing Summary in {report}")


def section_lines(lines: list[str], title: str) -> list[str]:
    start = None
    marker = f"| {title}"
    for index, line in enumerate(lines):
        if marker in line:
            start = index + 1
            break
    if start is None:
        raise SystemExit(f"could not find {title}")
    result: list[str] = []
    for line in lines[start:]:
        # Section headings start with "| <letter>".  Do not mistake the
        # decorative "| -----" line immediately below a heading for the next
        # section.
        if re.match(r"^\|\s+[A-Za-z]", line.lstrip()) and title not in line:
            break
        result.append(line)
    return result


def parse_clock_timing(
    report: Path, clock: str
) -> tuple[float, float, float, float]:
    lines = report.read_text(errors="replace").splitlines()
    for line in section_lines(lines, "Intra Clock Table"):
        fields = line.split()
        if not fields or fields[0] != clock or len(fields) < 7:
            continue
        try:
            return (
                float(fields[1]),
                float(fields[2]),
                float(fields[5]),
                float(fields[6]),
            )
        except ValueError:
            continue
    raise SystemExit(f"could not parse clock {clock!r} in Intra Clock Table")


def parse_clock_period(report: Path, clock: str) -> tuple[float, float]:
    lines = report.read_text(errors="replace").splitlines()
    for line in section_lines(lines, "Clock Summary"):
        fields = line.split()
        if not fields or fields[0] != clock or len(fields) < 3:
            continue
        try:
            return float(fields[-2]), float(fields[-1])
        except ValueError:
            continue
    raise SystemExit(f"could not parse clock {clock!r} in Clock Summary")


def parse_xclbin_scaled_clock(xclbin: Path, clock: str) -> float:
    if not xclbin.is_file() or xclbin.stat().st_size == 0:
        raise SystemExit(f"missing or empty xclbin: {xclbin}")
    try:
        result = subprocess.run(
            ["xclbinutil", "--quiet", "--input", str(xclbin), "--info"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"could not inspect xclbin clocks: {error}") from error
    in_scalable = False
    current_name = None
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped == "Scalable Clocks":
            in_scalable = True
            continue
        if in_scalable and stripped == "System Clocks":
            break
        name_match = re.fullmatch(r"Name:\s*(\S+)", stripped)
        if name_match:
            current_name = name_match.group(1)
            continue
        frequency_match = re.fullmatch(r"Frequency:\s*([0-9.]+)\s*MHz", stripped)
        if frequency_match and current_name == clock:
            return float(frequency_match.group(1))
    raise SystemExit(f"could not find scalable clock {clock!r} in {xclbin}")


def main() -> int:
    args = parse_args()
    if args.allow_auto_scale_xclbin is not None and args.clock is None:
        raise SystemExit("--allow-auto-scale-xclbin requires --clock")
    if args.clock is None:
        wns, tns, whs, ths = parse_design_summary(args.report)
    else:
        wns, tns, whs, ths = parse_clock_timing(args.report, args.clock)
    values = (wns, tns, whs, ths)
    if not all(math.isfinite(value) for value in values):
        raise SystemExit(f"non-finite routed timing values: {values}")
    print(
        "routed_timing_gate "
        f"clock={args.clock or 'DESIGN'} "
        f"WNS={wns:.3f} TNS={tns:.3f} WHS={whs:.3f} THS={ths:.3f}"
    )
    if whs < args.min_whs:
        raise SystemExit(
            "ROUTED_TIMING_GATE: FAIL "
            f"(required WHS>={args.min_whs:.3f})"
        )
    if wns < args.min_wns:
        if args.allow_auto_scale_xclbin is None:
            raise SystemExit(
                "ROUTED_TIMING_GATE: FAIL "
                f"(required WNS>={args.min_wns:.3f})"
            )
        period_ns, original_mhz = parse_clock_period(args.report, args.clock)
        scaled_mhz = parse_xclbin_scaled_clock(
            args.allow_auto_scale_xclbin, args.clock
        )
        # A setup slack of WNS means the worst data path takes period-WNS.
        # Scaling to at most its reciprocal makes that path non-negative.
        safe_mhz = 1000.0 / (period_ns - wns)
        # xclbinutil prints integral MHz, while Vivado computes tenths of MHz.
        tolerance_mhz = 0.5
        print(
            "routed_timing_auto_scale "
            f"clock={args.clock} original={original_mhz:.3f}MHz "
            f"safe_max={safe_mhz:.3f}MHz xclbin={scaled_mhz:.3f}MHz"
        )
        if scaled_mhz >= original_mhz or scaled_mhz > safe_mhz + tolerance_mhz:
            raise SystemExit(
                "ROUTED_TIMING_GATE: FAIL "
                "(xclbin scalable clock is not low enough for routed WNS)"
            )
        print("ROUTED_TIMING_GATE: PASS (xclbin auto-scaled)")
        return 0
    print("ROUTED_TIMING_GATE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
