#!/usr/bin/env python3
"""
Generate a horizontal bar chart that visualizes every cause-of-loss category
for Smith County corn claims (1994-2024).

The script uses only Python's standard library and renders an SVG file so we
avoid heavyweight plotting dependencies (which are not available in this
environment). Usage:

    python generate_cause_of_loss_chart.py \
        --input "2025-26 MTFC Scenario Dataset ... Smith Co.csv" \
        --output outputs/corn_cause_of_loss_all_causes.svg

By default the script reads the CSV located in the repository root and writes
the SVG into ./outputs/.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = REPO_ROOT / "2025-26 MTFC Scenario Dataset (updated 9_15_25)_1757954358.xlsx - Cause of Loss Smith Co.csv"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "corn_cause_of_loss_all_causes.svg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render cause-of-loss chart as SVG.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the Smith County cause-of-loss CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for the generated SVG (default: %(default)s)",
    )
    parser.add_argument("--start-year", type=int, default=1994, help="Inclusive start year filter.")
    parser.add_argument("--end-year", type=int, default=2024, help="Inclusive end year filter.")
    return parser.parse_args()


def load_cause_counts(
    csv_path: Path, start_year: int, end_year: int
) -> List[Tuple[str, float]]:
    """Return [(cause, policies_paid)] sorted descending by count."""
    counts: Dict[str, float] = defaultdict(float)
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            year = _safe_int(row.get("Year"))
            if year is None or not (start_year <= year <= end_year):
                continue
            policies = _safe_float(row.get("# of Policies Paid Out"))
            if policies is None:
                continue
            cause = (row.get("Cause of Loss") or "Unknown").strip() or "Unknown"
            counts[cause] += policies
    if not counts:
        raise ValueError("No cause-of-loss rows matched the provided filters.")
    return sorted(counts.items(), key=lambda kv: kv[1], reverse=True)


def _safe_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value.strip())
    except ValueError:
        return None


def _safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    cleaned = value.strip().replace(",", "")
    if not cleaned or cleaned in {"#DIV/0!", "?"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def build_svg(
    data: List[Tuple[str, float]],
    output_path: Path,
    start_year: int,
    end_year: int,
) -> None:
    labels = [cause for cause, _ in data]
    values = [count for _, count in data]
    total = sum(values)
    max_value = max(values)

    # Layout configuration
    char_px = 7  # rough estimate—keeps long labels readable
    longest_label = max(len(label) for label in labels)
    left_margin = max(260, 40 + char_px * longest_label)
    right_margin = 260
    top_margin = 90
    bottom_margin = 80
    bar_height = 22
    bar_gap = 10
    inner_width = 800
    scale = inner_width / max_value

    height = top_margin + bottom_margin + len(labels) * (bar_height + bar_gap)
    width = left_margin + inner_width + right_margin

    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    axis_color = "#1a1a1a"
    value_color = "#111111"
    subtitle_color = "#4d4d4d"
    grid_color = "#e0e0e0"

    svg_lines: List[str] = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}' "
        "font-family='Helvetica, Arial, sans-serif'>",
        f"  <rect x='0' y='0' width='{width}' height='{height}' fill='#fcfcfc' />",
        "  <style>",
        "    text { font-size: 14px; }",
        "    .title { font-size: 24px; font-weight: bold; fill: #111; }",
        "    .subtitle { font-size: 14px; fill: #555; }",
        "    .label { font-size: 13px; fill: #222; }",
        "  </style>",
        f"  <text class='title' x='{width/2}' y='{top_margin - 50}' text-anchor='middle'>"
        f"Corn Claims by Cause of Loss ({start_year}-{end_year})</text>",
    ]
    svg_lines.append(
        f"  <text class='subtitle' x='{width/2}' y='{top_margin - 26}' text-anchor='middle'>"
        "# of policies paid per cause</text>"
    )

    svg_lines.append(
        f"  <line x1='{left_margin}' y1='{top_margin-10}' x2='{left_margin}' y2='{height-bottom_margin+10}' "
        f"stroke='{axis_color}' stroke-width='1.25' />"
    )
    svg_lines.append(
        f"  <line x1='{left_margin}' y1='{height-bottom_margin}' x2='{width-right_margin}' y2='{height-bottom_margin}' "
        f"stroke='{axis_color}' stroke-width='1.25' />"
    )

    tick_count = 6
    for i in range(tick_count):
        value = max_value * i / (tick_count - 1)
        x = left_margin + value * scale
        svg_lines.append(f"  <line x1='{x}' y1='{top_margin-10}' x2='{x}' y2='{height-bottom_margin}' stroke='{grid_color}' />")
        svg_lines.append(
            f"  <text x='{x}' y='{height-bottom_margin + 28}' text-anchor='middle' fill='{axis_color}'>{int(value):,}</text>"
        )

    for idx, (label, value) in enumerate(data):
        y = top_margin + idx * (bar_height + bar_gap)
        color = palette[idx % len(palette)]
        width_px = value * scale
        svg_lines.append(
            f"  <rect x='{left_margin}' y='{y}' width='{width_px}' height='{bar_height}' fill='{color}' rx='4' ry='4' />"
        )
        svg_lines.append(
            f"  <text class='label' x='{left_margin - 16}' y='{y + bar_height / 2 + 5}' text-anchor='end'>{label}</text>"
        )
        svg_lines.append(
            f"  <text x='{left_margin + width_px + 16}' y='{y + bar_height / 2 + 5}' fill='{value_color}'>"
            f"{value:,.0f} ({value/total:.1%})</text>"
        )

    svg_lines.append(
        f"  <text class='subtitle' x='{width/2}' y='{height-30}' text-anchor='middle'>Total policies counted: {total:,.0f}</text>"
    )
    svg_lines.append("</svg>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg_lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    counts = load_cause_counts(args.input, args.start_year, args.end_year)
    build_svg(counts, args.output, args.start_year, args.end_year)
    print(f"Wrote chart with {len(counts)} causes to {args.output}")


if __name__ == "__main__":
    main()

