#!/usr/bin/env python3
"""
Calculate the average annual insurance payout per policy for drought claims
in Smith County, Iowa (1994-2024) for MTFC Scenario Quest Question #22.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def main():
    # Load the data
    data_path = Path(__file__).parent / "2025-26 MTFC Scenario Dataset (updated 9_15_25)_1757954358.xlsx - Cause of Loss Smith Co.csv"
    df = pd.read_csv(data_path)

    # Filter for drought claims only (1994-2024)
    df = df[df['Year'].between(1994, 2024)]
    drought_df = df[df['Cause of Loss'] == 'Drought'].copy()

    # Clean numeric columns
    drought_df['# of Policies Paid Out'] = pd.to_numeric(
        drought_df['# of Policies Paid Out'].astype(str).str.replace(',', ''), 
        errors='coerce'
    )
    drought_df['Avg amout paid out per policy'] = (
        drought_df['Avg amout paid out per policy']
        .astype(str)
        .str.replace('$', '', regex=False)
        .str.replace(',', '', regex=False)
    )
    drought_df['Avg amout paid out per policy'] = pd.to_numeric(
        drought_df['Avg amout paid out per policy'], 
        errors='coerce'
    )

    # Drop rows with missing values
    drought_df = drought_df.dropna(subset=['# of Policies Paid Out', 'Avg amout paid out per policy'])

    # Calculate total payouts and total policies
    drought_df['Total Payout'] = drought_df['# of Policies Paid Out'] * drought_df['Avg amout paid out per policy']

    total_policies = drought_df['# of Policies Paid Out'].sum()
    total_payout = drought_df['Total Payout'].sum()
    years = 31  # 1994-2024

    # Average payout per policy
    avg_payout_per_policy = total_payout / total_policies

    # Annual averages
    annual_avg_payout = total_payout / years
    annual_avg_policies = total_policies / years

    # Print results
    print("=" * 60)
    print("#22: AVERAGE ANNUAL INSURANCE PAYOUT DUE TO DROUGHT")
    print("=" * 60)
    print()
    print("Drought Claims Summary (1994-2024):")
    print(f"  Total drought policies paid out: {total_policies:,.0f}")
    print(f"  Total drought payouts: ${total_payout:,.2f}")
    print(f"  Years in dataset: {years}")
    print()
    print("Calculations:")
    print(f"  Average payout per policy: ${total_payout:,.2f} / {total_policies:,.0f}")
    print(f"                           = ${avg_payout_per_policy:,.2f}")
    print()
    print(f"  Annual average policies paid: {annual_avg_policies:,.1f}")
    print(f"  Annual average total payout: ${annual_avg_payout:,.2f}")
    print()
    print("-" * 60)
    print(f"ANSWER: Average insurance payout per policy for drought = ${avg_payout_per_policy:,.2f}")
    print("-" * 60)

    # Context for Farmer Jones
    planting_cost = 189097.95
    print()
    print("Context for Farmer Jones:")
    print(f"  Farmer Jones' total planting cost (from #7): ${planting_cost:,.2f}")
    print(f"  Average drought payout per policy: ${avg_payout_per_policy:,.2f}")
    print(f"  Payout as % of planting cost: {(avg_payout_per_policy / planting_cost) * 100:.1f}%")
    print()
    
    return avg_payout_per_policy, annual_avg_payout, total_policies


if __name__ == "__main__":
    main()

