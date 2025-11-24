# MTFC - Smith County Corn Loss Claims Analysis

Analysis of corn crop insurance loss claims in Smith County (1994-2024) for the MTFC Scenario Quest.

## Project Overview

This repository contains data analysis and visualizations for understanding:
- Average corn production costs per acre and per bushel (2016-2025)
- Corn price trends across marketing years
- Causes of loss for insurance claims (1994-2024)
- Linear regression analysis of top 3 loss causes

## Files

### Data Files
- `2025-26 MTFC Scenario Dataset (updated 9_15_25)_1757954358.xlsx - Cause of Loss Smith Co.csv` - Loss claims data
- `2025-26 MTFC Scenario Dataset (updated 9_15_25)_1757954358.xlsx - Corn Harvest Prices.csv` - Historical corn prices
- `2025-26 MTFC Scenario Dataset (updated 9_15_25)_1757954358.xlsx - Corn Planting Costs.csv` - Production cost data

### Analysis Scripts
- `generate_cause_of_loss_chart.py` - Generates comprehensive bar chart of all 31 loss causes
- `analyze_top3_loss_trends.py` - Performs linear regression on top 3 loss causes (1994-2024)

### Output Visualizations
- `outputs/corn_cause_of_loss_all_causes.svg` - Complete breakdown of loss causes
- `outputs/top3_loss_trend_regressions.svg` - Time series with regression lines for drought, price decline, and excess moisture

## Setup

Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Generate cause of loss chart:
```bash
python generate_cause_of_loss_chart.py
```

Run regression analysis:
```bash
python analyze_top3_loss_trends.py
```

## Key Findings

### Top 3 Loss Causes (1994-2024)
1. **Drought** - 2,781 policies (38.2%)
2. **Decline in Price** - 2,233 policies (30.6%)
3. **Excess Moisture/Precipitation/Rain** - 972 policies (13.3%)

### Regression Results
- **Drought**: y = 5.29x - 10,534, r = 0.28, p = 0.123
- **Decline in Price**: y = 4.72x - 9,401, r = 0.23, p = 0.220
- **Excess Moisture**: y = 1.92x - 3,833, r = 0.28, p = 0.130

All trends show weak positive correlations, indicating that year-to-year variation is driven more by catastrophic events than steady time trends.

## Requirements

- Python 3.9+
- numpy >= 1.24.0
- pandas >= 2.0.0
- scipy >= 1.10.0

## License

Educational project for MTFC Scenario Quest 2025-26.

