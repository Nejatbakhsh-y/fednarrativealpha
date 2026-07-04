# FedNarrativeAlpha

FedNarrativeAlpha is a reproducible alpha-research pipeline for liquid ETF markets.

The project tests whether macroeconomic indicators and Federal Reserve narrative embeddings improve next-month ETF allocation under realistic walk-forward validation, transaction costs, and risk constraints.

The system follows a full research chain:

```text
data → hypothesis → feature construction → model → validation → backtest → risk optimization → reproducible report
```

## Project Purpose

This is a research and model-validation project. Results are evaluated out-of-sample and after transaction-cost assumptions. The project is not investment advice.

The goal is not to claim that the model predicts the stock market. Instead, the project demonstrates how to design, validate, and report a quantitative research pipeline using proper controls for financial time-series modeling.

## Research Question

Does adding Federal Reserve narrative information to price and macroeconomic features improve next-month ETF allocation performance after realistic validation, transaction costs, and portfolio constraints?

## Core Hypothesis

Federal Reserve communications may contain information about inflation, growth, interest rates, liquidity, and macroeconomic uncertainty. Sentence-transformer embeddings are used to convert Federal Reserve text into numerical features, which are then tested against non-text baselines.

The main model comparison is:

```text
price features
vs.
price + macro features
vs.
price + macro + Federal Reserve narrative features
```

## Pipeline Overview

The project is structured as a complete quantitative research workflow:

1. Download ETF price data.
2. Download macroeconomic data from FRED.
3. Build monthly price and macro features.
4. Create next-month ETF return targets.
5. Add Federal Reserve narrative embeddings.
6. Train baseline and machine-learning models.
7. Use walk-forward validation.
8. Convert predictions into ETF allocation signals.
9. Apply transaction-cost assumptions.
10. Run portfolio backtests.
11. Apply risk-constrained optimization.
12. Generate reproducible research reports and model documentation.

## Validation Design

The project uses walk-forward validation instead of random train-test splitting.

This is important because financial time-series modeling must preserve chronological order. Training data must always precede test data, and all features must be available before the prediction date.

The validation design is intended to reduce:

* Look-ahead bias
* Data leakage
* Overfitting from random splits
* Unrealistic backtest assumptions

## Backtest Design

The backtest evaluates ETF allocation rules using predicted next-month returns. The portfolio construction layer converts model predictions into ETF weights and then applies transaction-cost assumptions.

The project compares the full narrative-enhanced model against simpler benchmarks, including:

* SPY benchmark
* 60/40 portfolio
* Equal-weight ETF universe
* Momentum-only strategy
* Macro-only strategy
* Price + macro model
* Price + macro + Federal Reserve text model

## Risk Controls

The portfolio optimization layer supports practical constraints such as:

* Long-only weights
* Maximum single-ETF allocation
* Minimum selected-ETF allocation
* Turnover limits
* Target volatility controls
* Optional asset-class exposure limits

These controls make the research process closer to realistic portfolio construction and model-validation practice.

## Expected Outputs

The repository is designed to produce the following main outputs:

```text
data/processed/master_modeling_dataset.parquet
results/selected_features.csv
results/walk_forward_predictions.csv
results/model_metrics.json
results/feature_importance.csv
results/portfolio_weights.csv
results/backtest_results.csv
results/backtest_summary.json
reports/final_research_report.md
reports/model_card.md
reports/figures/cumulative_returns.png
reports/figures/drawdown_curve.png
reports/figures/feature_importance.png
reports/figures/walk_forward_rank_ic.png
reports/figures/portfolio_weights_over_time.png
```

## Repository Structure

```text
FedNarrativeAlpha/
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── results/
│
├── reports/
│   └── figures/
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── backtest/
│   └── reporting/
│
├── tests/
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Main Modules

### `src/data/`

Data ingestion and cleaning.

### `src/features/`

Price, macroeconomic, and Federal Reserve text-feature construction.

### `src/models/`

Baseline models, machine-learning models, feature selection, and walk-forward validation.

### `src/backtest/`

Signal conversion, transaction costs, portfolio optimization, and backtest performance metrics.

### `src/reporting/`

Reproducible research report and model-card generation.

### `tests/`

Validation tests for no-lookahead controls, feature dates, portfolio constraints, and backtest accounting.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the machine-learning models:

```bash
python src/models/train_ml_models.py
```

Run walk-forward validation:

```bash
python src/models/walk_forward_validation.py
```

Convert predictions into portfolio signals:

```bash
python src/backtest/signal_to_portfolio.py
```

Apply transaction costs:

```bash
python src/backtest/transaction_costs.py
```

Run the backtest engine:

```bash
python src/backtest/backtest_engine.py
```

Generate the final research report:

```bash
python src/reporting/make_research_report.py
```

Run validation tests:

```bash
pytest
```

## Important Limitations

This project does not claim to predict the stock market.

This project does not guarantee profitable trading.

This project is not an investment strategy.

This project is not investment advice.

The purpose is to demonstrate a disciplined research process for testing whether macroeconomic indicators and Federal Reserve narrative features add measurable value under realistic validation and portfolio-construction assumptions.

## Model Governance Notes

The project includes model-risk and research-governance controls such as:

* Out-of-sample validation
* Chronological train/test separation
* No-lookahead checks
* Transaction-cost assumptions
* Benchmark comparisons
* Model-card documentation
* Reproducible reporting
* Explicit limitations and failure cases

## Disclaimer

This repository is for educational, research, and model-validation purposes only. Nothing in this repository should be interpreted as financial advice, investment advice, or a recommendation to buy or sell any security.
