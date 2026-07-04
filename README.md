# FedNarrativeAlpha: GenAI-Assisted Alpha Research Pipeline for Liquid ETF Markets

FedNarrativeAlpha is a reproducible quantitative research project that tests whether Federal Reserve communication, macroeconomic indicators, and ETF market behavior can be combined into useful alpha signals for liquid ETF markets.

The project is designed as a full research pipeline rather than a simple prediction exercise. It connects data ingestion, feature engineering, information-theoretic signal selection, walk-forward validation, transaction-cost-aware backtesting, risk-constrained portfolio construction, model-risk controls, and reproducible reporting.

## Project Objective

The objective is to evaluate whether Federal Reserve narrative information adds incremental out-of-sample value beyond price and macroeconomic features when allocating across liquid ETFs.

The central research question is:

> Do Federal Reserve communication features improve next-month ETF allocation performance after realistic validation, transaction costs, and portfolio constraints?

The project emphasizes disciplined research design. The final Sharpe ratio is not the only measure of success. The main goal is to show whether a proposed signal survives realistic out-of-sample testing.

## Research Workflow

The project follows this research chain:

```text
data
-> hypothesis
-> feature construction
-> information-theoretic signal selection
-> model training
-> walk-forward validation
-> portfolio signal construction
-> transaction-cost-aware backtesting
-> portfolio optimization
-> reporting
-> model-risk documentation
```

## Core Components

### 1. Data Ingestion

The project uses public market and macroeconomic data sources, including:

* Liquid ETF price data
* FRED macroeconomic indicators
* Federal Reserve communication text, such as FOMC statements

Raw data are not committed to the repository. The project is structured so that data can be regenerated through reproducible scripts.

### 2. Feature Engineering

The project builds monthly modeling features from:

* ETF price and return history
* Momentum and volatility indicators
* Macro indicators
* Macro changes and lagged macro variables
* Federal Reserve text embeddings and narrative features

The target variable is next-month ETF return.

### 3. Hypothesis Registration

The project includes a hypothesis-driven research structure. Instead of blindly fitting models, the workflow documents the expected relationship between feature groups and future ETF returns.

This helps separate genuine research logic from post-hoc model selection.

### 4. Information-Theoretic Signal Selection

The project uses information-theoretic feature selection to identify features with potential predictive content while reducing redundancy.

The feature-selection layer is designed to evaluate:

* Mutual information with the target
* Redundancy across predictors
* Stability of selected signals
* Incremental value of Fed narrative features

### 5. Machine Learning Models

The project compares several model families, including:

* Baseline models
* Ridge regression
* Elastic Net
* Random Forest
* Gradient Boosting
* XGBoost
* LightGBM

The models are evaluated across multiple feature sets:

* Price features only
* Price plus macro features
* Price plus macro plus Fed text features
* Information-theoretically selected features

This comparison is central to the project because it tests whether the GenAI/text layer adds incremental value beyond simpler alternatives.

### 6. Walk-Forward Validation

The project uses chronological walk-forward validation rather than random train-test splits.

The validation design includes:

* Rolling or expanding training windows
* One-month-ahead test periods
* Chronological train/test separation
* Feature-date integrity checks
* No-lookahead controls

This is essential for financial time-series research because random splits can create misleading results and data leakage.

### 7. Portfolio Signal Construction

Model predictions are converted into implementable portfolio signals.

The basic portfolio rule is:

* Rank ETFs by predicted next-month return
* Select the top-ranked ETFs
* Allocate equal weights or optimized weights
* Rebalance monthly

The project stores portfolio weights for inspection and reproducibility.

### 8. Transaction-Cost-Aware Backtesting

The backtest includes realistic transaction-cost assumptions.

The project evaluates multiple cost scenarios:

* Low cost: 5 basis points per one-way trade
* Medium cost: 10 basis points per one-way trade
* High cost: 25 basis points per one-way trade

The backtest separates:

* Gross returns
* Turnover
* Transaction costs
* Net returns
* Transaction-cost drag

This helps determine whether a signal remains useful after trading costs.

### 9. Risk-Constrained Portfolio Optimization

The project includes portfolio optimization with practical constraints, such as:

* Long-only weights
* Maximum single-ETF exposure
* Minimum selected ETF weight
* Maximum turnover
* Target volatility
* Optional asset-class exposure limits

The objective is to connect model forecasts to realistic portfolio construction.

### 10. Model-Risk Controls

The project includes tests and documentation for model-risk control.

Examples include:

* No feature date can be later than the prediction date
* Walk-forward training dates must precede test dates
* Portfolio weights must sum to one
* ETF weights must satisfy maximum allocation limits
* Transaction costs must reduce gross returns
* Backtest accounting must be internally consistent

These controls help make the project credible as a research system.

## Expected Repository Structure

```text
fednarrativealpha/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── docs/
│   └── edgestream_positioning.md
├── notebooks/
├── reports/
│   ├── final_research_report.md
│   ├── model_card.md
│   └── figures/
├── results/
│   ├── selected_features.csv
│   ├── walk_forward_predictions.csv
│   ├── model_metrics.json
│   ├── feature_importance.csv
│   ├── portfolio_weights.csv
│   ├── optimization_diagnostics.csv
│   ├── backtest_results.csv
│   └── backtest_summary.json
├── scripts/
├── src/
│   ├── backtest/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── reporting/
├── tests/
├── .env.example
├── .gitignore
├── pyproject.toml
├── README.md
└── requirements.txt
```

## Expected Outputs

A complete run of the project should produce the following research outputs:

```text
data/processed/master_modeling_dataset.parquet
results/selected_features.csv
results/walk_forward_predictions.csv
results/model_metrics.json
results/feature_importance.csv
results/portfolio_weights.csv
results/optimization_diagnostics.csv
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

## How to Run

From the project root, install the required packages:

```powershell
pip install -r requirements.txt
```

Then run the main pipeline scripts in sequence.

```powershell
python src\data\download_market_data.py
python src\data\download_macro_data.py
python src\features\build_price_macro_features.py
python src\features\build_targets.py
python src\features\select_information_features.py
python src\models\train_baselines.py
python src\models\train_ml_models.py
python src\models\walk_forward_validation.py
python src\backtest\signal_to_portfolio.py
python src\backtest\transaction_costs.py
python src\backtest\portfolio_optimizer.py
python src\backtest\backtest_engine.py
python src\reporting\make_research_report.py
```

Then run the test suite:

```powershell
pytest
```

## Validation Philosophy

This project is intentionally conservative about model evaluation.

It avoids relying on in-sample performance. Instead, it emphasizes:

* Chronological validation
* Out-of-sample prediction
* Benchmark comparison
* Transaction-cost realism
* Portfolio constraints
* No-lookahead testing
* Transparent reporting of limitations

The goal is not to present an overfit trading strategy. The goal is to demonstrate a rigorous research process for testing whether a market signal is robust.

## Why This Project Is Relevant to Edgestream

This project is designed to demonstrate rigorous quantitative research engineering: hypothesis-driven alpha research, time-series validation, information-theoretic feature selection, transaction-cost-aware backtesting, portfolio construction, model-risk controls, and reproducible reporting.

The main value is not whether the final Sharpe ratio is high. The main value is that the repository shows a complete research process for testing whether a signal survives realistic out-of-sample evaluation.

See: [Why FedNarrativeAlpha Is Strong for Edgestream](docs/edgestream_positioning.md)

## Key Skills Demonstrated

This repository demonstrates:

* Quantitative research design
* Alpha signal testing
* Financial time-series validation
* Machine learning for noisy market data
* Information-theoretic feature selection
* Portfolio construction
* Transaction-cost-aware backtesting
* Risk-constrained optimization
* Reproducible Python engineering
* Model validation and governance documentation
* GenAI-assisted research workflow design
* Clear research communication

## Limitations

This project is for research and portfolio demonstration purposes only.

Important limitations include:

* Public data may be revised, delayed, or incomplete
* ETF returns are noisy and difficult to predict
* Backtest results may not generalize to live trading
* Transaction-cost assumptions are simplified
* Model performance may be regime-dependent
* Federal Reserve text features may not provide a stable incremental signal
* The project does not represent a production trading system

## Disclaimer

This repository is not investment advice. It is a research and engineering project designed to demonstrate quantitative research methodology, reproducible modeling, backtesting discipline, and model-risk documentation.

No results in this repository should be interpreted as a recommendation to buy, sell, or trade any financial instrument.
