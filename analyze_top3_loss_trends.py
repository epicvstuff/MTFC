#!/usr/bin/env python3
"""
Analyze Smith County corn loss claims (1994-2024) for the top three causes.

The script performs a simple linear regression of policy counts vs. year for
each cause and generates:
  1. Console output listing each regression equation and correlation coefficient
  2. An SVG line chart plotting yearly counts and regression lines for all
     three causes on the same axes

Example usage (default paths are pre-filled for the current repository):

    python analyze_top3_loss_trends.py

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    REPO_ROOT
    / "2025-26 MTFC Scenario Dataset (updated 9_15_25)_1757954358.xlsx - Cause of Loss Smith Co.csv"
)
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "top3_loss_trend_regressions.svg"


@dataclass
class RegressionResult:
    cause: str
    slope: float
    intercept: float
    r_value: float
    p_value: float
    yearly_counts: List[Tuple[int, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Linear regression of top 3 loss causes.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="CSV path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="SVG output path")
    parser.add_argument("--start-year", type=int, default=1994, help="inclusive start year")
    parser.add_argument("--end-year", type=int, default=2024, help="inclusive end year")
    return parser.parse_args()


def load_and_aggregate_data(csv_path: Path, start_year: int, end_year: int) -> pd.DataFrame:
    """Load CSV and aggregate policies by cause and year."""
    df = pd.read_csv(
        csv_path,
        usecols=["Year", "Cause of Loss", "# of Policies Paid Out"],
        dtype={"Year": "Int64", "Cause of Loss": str, "# of Policies Paid Out": str},
    )
    
    # Clean and filter
    df = df.dropna(subset=["Year", "Cause of Loss"])
    df = df[(df["Year"] >= start_year) & (df["Year"] <= end_year)]
    
    # Convert policies to numeric, coercing errors
    df["# of Policies Paid Out"] = (
        df["# of Policies Paid Out"]
        .str.replace(",", "", regex=False)
        .replace({"#DIV/0!": np.nan, "?": np.nan, "": np.nan})
    )
    df["# of Policies Paid Out"] = pd.to_numeric(df["# of Policies Paid Out"], errors="coerce")
    df = df.dropna(subset=["# of Policies Paid Out"])
    
    # Aggregate by cause and year
    aggregated = (
        df.groupby(["Cause of Loss", "Year"])["# of Policies Paid Out"]
        .sum()
        .reset_index()
    )
    
    return aggregated


def compute_top3_regressions(
    df: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> List[RegressionResult]:
    """Identify top 3 causes by total policies and compute linear regressions."""
    # Get top 3 causes
    totals = df.groupby("Cause of Loss")["# of Policies Paid Out"].sum().sort_values(ascending=False)
    top3_causes = totals.head(3).index.tolist()
    
    years = np.arange(start_year, end_year + 1)
    results: List[RegressionResult] = []
    
    for cause in top3_causes:
        cause_data = df[df["Cause of Loss"] == cause].set_index("Year")["# of Policies Paid Out"]
        
        # Fill missing years with 0
        yearly_counts_series = cause_data.reindex(years, fill_value=0.0)
        yearly_counts = list(zip(years, yearly_counts_series.values))
        
        # Perform linear regression using scipy
        slope, intercept, r_value, p_value, std_err = stats.linregress(years, yearly_counts_series.values)
        
        results.append(
            RegressionResult(
                cause=cause,
                slope=slope,
                intercept=intercept,
                r_value=r_value,
                p_value=p_value,
                yearly_counts=yearly_counts,
            )
        )
    
    return results


def build_svg(results: List[RegressionResult], output_path: Path, start_year: int, end_year: int) -> None:
    """Generate SVG line chart with regression lines."""
    years = np.arange(start_year, end_year + 1)
    max_y = max(value for result in results for _, value in result.yearly_counts)

    width = 1200
    height = 640
    left_margin = 90
    right_margin = 260
    top_margin = 80
    bottom_margin = 90
    plot_width = width - left_margin - right_margin
    plot_height = height - top_margin - bottom_margin

    def x_for_year(year: int) -> float:
        return left_margin + (year - start_year) / (end_year - start_year) * plot_width

    def y_for_val(value: float) -> float:
        if max_y == 0:
            return top_margin + plot_height
        return top_margin + (1 - value / max_y) * plot_height

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    svg = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}' "
        "font-family='Helvetica, Arial, sans-serif'>",
        "  <rect x='0' y='0' width='{width}' height='{height}' fill='#ffffff' />".format(
            width=width, height=height
        ),
        "  <style>",
        "    text { font-size: 14px; }",
        "    .title { font-size: 22px; font-weight: bold; fill: #111; }",
        "    .subtitle { font-size: 14px; fill: #555; }",
        "    .legend { font-size: 14px; }",
        "  </style>",
        f"  <text class='title' x='{width/2}' y='36' text-anchor='middle'>Top 3 Loss Claim Trends (1994-2024)</text>",
        f"  <text class='subtitle' x='{width/2}' y='58' text-anchor='middle'>Yearly policies paid and linear regression lines</text>",
    ]

    # Axes
    svg.append(
        f"  <line x1='{left_margin}' y1='{top_margin - 10}' x2='{left_margin}' y2='{height - bottom_margin + 10}' "
        "stroke='#333' stroke-width='1.2' />"
    )
    svg.append(
        f"  <line x1='{left_margin}' y1='{height - bottom_margin}' x2='{width - right_margin}' y2='{height - bottom_margin}' "
        "stroke='#333' stroke-width='1.2' />"
    )

    # Y ticks
    for i in range(0, 6):
        value = max_y * i / 5
        y = y_for_val(value)
        svg.append(
            f"  <line x1='{left_margin}' y1='{y}' x2='{width - right_margin}' y2='{y}' stroke='#e0e0e0' />"
        )
        svg.append(
            f"  <text x='{left_margin - 10}' y='{y + 5}' text-anchor='end'>{value:,.0f}</text>"
        )

    # X ticks
    step = max(1, (end_year - start_year) // 6)
    for year in range(start_year, end_year + 1, step):
        x = x_for_year(year)
        svg.append(f"  <line x1='{x}' y1='{top_margin - 5}' x2='{x}' y2='{height - bottom_margin}' stroke='#e0e0e0' />")
        svg.append(
            f"  <text x='{x}' y='{height - bottom_margin + 28}' text-anchor='middle'>{year}</text>"
        )

    # Plot series
    for idx, result in enumerate(results):
        color = colors[idx % len(colors)]
        points = " ".join(
            f"{x_for_year(year):.2f},{y_for_val(count):.2f}"
            for year, count in result.yearly_counts
        )
        svg.append(
            f"  <polyline fill='none' stroke='{color}' stroke-width='2' points='{points}' />"
        )
        for year, count in result.yearly_counts:
            x = x_for_year(year)
            y = y_for_val(count)
            svg.append(
                f"  <circle cx='{x:.2f}' cy='{y:.2f}' r='2.5' fill='{color}' />"
            )

        # Regression line
        reg_y_start = result.slope * start_year + result.intercept
        reg_y_end = result.slope * end_year + result.intercept
        svg.append(
            f"  <line x1='{x_for_year(start_year):.2f}' y1='{y_for_val(reg_y_start):.2f}' "
            f"x2='{x_for_year(end_year):.2f}' y2='{y_for_val(reg_y_end):.2f}' "
            f"stroke='{color}' stroke-dasharray='6,4' stroke-width='2' />"
        )

    # Legend
    legend_x = width - right_margin + 10
    legend_y = top_margin
    for idx, result in enumerate(results):
        color = colors[idx % len(colors)]
        svg.append(
            f"  <rect x='{legend_x}' y='{legend_y + idx * 34 - 13}' width='16' height='3' fill='{color}' />"
        )
        svg.append(
            f"  <text class='legend' x='{legend_x + 24}' y='{legend_y + idx * 34}'>{result.cause}</text>"
        )
        svg.append(
            f"  <text class='legend' x='{legend_x + 24}' y='{legend_y + idx * 34 + 16}' fill='#444'>"
            f"y = {result.slope:.2f}x + {result.intercept:.0f}, r = {result.r_value:.2f}, p = {result.p_value:.3f}</text>"
        )

    svg.append(f"  <text class='subtitle' x='{width/2}' y='{height - 20}' text-anchor='middle'>Source: Smith County corn claims (RMA)</text>")
    svg.append("</svg>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    args = parse_args()
    df = load_and_aggregate_data(args.input, args.start_year, args.end_year)
    results = compute_top3_regressions(df, args.start_year, args.end_year)
    build_svg(results, args.output, args.start_year, args.end_year)

    print("Regression results:")
    for result in results:
        print(
            f"- {result.cause}: y = {result.slope:.3f} * year + {result.intercept:.2f}, "
            f"r = {result.r_value:.3f}, p = {result.p_value:.4f}"
        )


if __name__ == "__main__":
    main()

