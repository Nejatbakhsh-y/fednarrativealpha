# FedNarrativeAlpha: GenAI-Assisted Alpha Research Pipeline for Liquid ETF Markets

FedNarrativeAlpha is a reproducible quantitative research project that studies whether Federal Reserve communication, macroeconomic conditions, and ETF market behavior can be combined into testable alpha signals for liquid ETF markets.

## Project Objective

The objective is to build a research pipeline that connects:

1. ETF market data
2. Federal Reserve narrative and macro signals
3. Feature engineering
4. Time-series validation
5. Transaction-cost-aware backtesting
6. Alpha signal reporting
7. GenAI-assisted research summaries

## Repository Structure

```text
fednarrativealpha/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── docs/
├── notebooks/
├── reports/
│   └── figures/
├── scripts/
├── src/
│   └── fednarrativealpha/
├── tests/
├── .env.example
├── .gitignore
├── pyproject.toml
├── README.md
└── requirements.txt
```

## Planned Research Modules

- ETF universe construction
- Market and macro data ingestion
- Federal Reserve communication collection
- Narrative feature engineering
- Signal construction
- Walk-forward validation
- Backtest evaluation
- Transaction cost analysis
- Model-card documentation
- GenAI-assisted research memo generation

## Target Outputs

The final repository will include:

```text
data/processed/etf_features.csv
data/processed/fed_narrative_features.csv
data/processed/modeling_dataset.csv
reports/model_metrics.json
reports/backtest_summary.csv
reports/alpha_research_memo.md
docs/model_card.md
```

## Technical Stack

- Python
- pandas
- NumPy
- scikit-learn
- statsmodels
- yfinance
- pandas-datareader
- matplotlib
- plotly
- Jupyter
- VS Code
- GitHub

## Status

Initial project setup in progress.