# FedNarrativeAlpha: GenAI-Assisted Alpha Research Pipeline for Liquid ETF Markets

FedNarrativeAlpha is a reproducible quantitative research project that studies whether Federal Reserve communication, macroeconomic conditions, and ETF market behavior can be combined into testable alpha signals for liquid ETF markets.

The project is designed as a practical research pipeline rather than a production trading system. Its purpose is to demonstrate a complete quantitative research workflow:

```text
public data
→ research hypothesis
→ feature construction
→ information-theoretic signal selection
→ walk-forward validation
→ transaction-cost-aware portfolio backtesting
→ risk-constrained allocation
→ reproducible research reporting
```

## Final Recommended Scope

FedNarrativeAlpha is built as a GenAI-assisted alpha research pipeline for liquid ETF markets.

The final project scope is intentionally practical and research-oriented. The goal is not to claim a production trading strategy, but to show a disciplined, reproducible alpha research workflow using public market data, macroeconomic indicators, Federal Reserve communication features, machine learning, GenAI-assisted hypothesis documentation, and realistic model validation.

The main evaluation question is:

> Does a full price + macro + Federal Reserve narrative model improve out-of-sample ETF allocation results relative to simpler baselines after accounting for transaction costs and risk constraints?

## Required Final Outputs

The repository is designed to produce the following final research artifacts:

```text
results/selected_features.csv
results/walk_forward_predictions.csv
results/model_metrics.json
results/feature_importance.csv
results/portfolio_weights.csv
results/backtest_results.csv
results/backtest_summary.json
reports/final_research_report.md
reports/model_card.md
```

These files document the full path from raw public data to validated model predictions, portfolio weights, backtest results, and model-risk documentation.

## Why This Project Is Strong

This project is suitable for quantitative research, systematic investing, and AI-assisted financial research roles because it demonstrates:

* liquid financial market data analysis
* public macroeconomic data integration
* Federal Reserve communication feature construction
* GenAI-assisted research documentation
* information-theoretic feature selection
* chronological walk-forward validation
* transaction-cost-aware backtesting
* portfolio optimization
* model-risk controls
* reproducible reporting
* model-card documentation

The most important objective is not to maximize a backtest metric. The main objective is to show that an alpha research idea can be converted into a clean, testable, reproducible, and honestly evaluated quantitative research system.

## Project Objective

The objective is to build a full research pipeline that connects:

1. ETF market data
2. Federal Reserve narrative and macroeconomic signals
3. Feature engineering
4. Information-theoretic feature selection
5. Baseline and machine-learning models
6. Walk-forward validation
7. Portfolio signal construction
8. Transaction-cost-aware backtesting
9. Risk-constrained portfolio optimization
10. Reproducible research reporting
11. Model-card documentation

## Research Design

FedNarrativeAlpha tests whether Federal Reserve communication features add predictive value beyond standard price and macroeconomic features.

The project compares several model families and feature sets:

```text
price-only model
macro-only model
price + macro model
price + macro + Federal Reserve narrative model
```

The research question is evaluated using out-of-sample walk-forward validation and portfolio backtesting after transaction costs.

## Data Sources

The project is based on public data sources.

Expected data groups include:

* liquid ETF price data
* macroeconomic indicators from public sources such as FRED
* Federal Reserve statements, speeches, or policy communication text
* derived monthly price, macro, and narrative features

Raw data files should not be committed to GitHub if they are large or automatically reproducible. Processed modeling outputs and final reports may be tracked when lightweight and useful for review.

## Repository Structure

```text
FedNarrativeAlpha/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── docs/
├── notebooks/
├── reports/
│   ├── figures/
│   ├── final_research_report.md
│   └── model_card.md
├── results/
│   ├── selected_features.csv
│   ├── walk_forward_predictions.csv
│   ├── model_metrics.json
│   ├── feature_importance.csv
│   ├── portfolio_weights.csv
│   ├── backtest_results.csv
│   └── backtest_summary.json
├── scripts/
├── src/
│   ├── backtest/
│   │   ├── backtest_engine.py
│   │   ├── portfolio_optimizer.py
│   │   ├── signal_to_portfolio.py
│   │   └── transaction_costs.py
│   ├── data/
│   ├── features/
│   ├── models/
│   │   ├── train_baselines.py
│   │   └── walk_forward_validation.py
│   └── reporting/
│       └── make_research_report.py
├── tests/
│   ├── test_backtest_accounting.py
│   ├── test_feature_dates.py
│   ├── test_no_lookahead.py
│   └── test_portfolio_constraints.py
├── .env.example
├── .gitignore
├── pyproject.toml
├── README.md
└── requirements.txt
```

## Core Pipeline

The intended pipeline is:

```text
1. Download public ETF price data
2. Download public macroeconomic indicators
3. Scrape or collect Federal Reserve communication text
4. Build monthly price, macro, and text-derived features
5. Create next-month ETF return targets
6. Select features using information-theoretic methods
7. Train baseline and machine-learning models
8. Run walk-forward validation
9. Convert model predictions into ETF portfolio weights
10. Apply transaction-cost assumptions
11. Run portfolio backtests
12. Generate summary metrics, reports, and model-card documentation
```

## Validation Design

The project uses chronological validation rather than random train-test splitting.

The walk-forward validation design uses:

* historical training windows
* forward test periods
* no future information in features
* no random shuffling of time-series observations
* feature-date checks
* test-period predictions generated only from prior data

This is essential because financial time series are highly sensitive to lookahead bias.

## Backtest Design

The backtest converts monthly model predictions into ETF portfolio weights.

The portfolio construction logic is based on:

* ranking ETFs by predicted next-month return
* selecting top-ranked ETFs
* assigning portfolio weights
* applying transaction costs
* calculating net returns
* comparing against benchmarks

The backtest is intended to report:

* annualized return
* annualized volatility
* Sharpe ratio
* Sortino ratio
* maximum drawdown
* Calmar ratio
* monthly turnover
* hit rate
* best month
* worst month
* transaction-cost drag

## Transaction Cost Scenarios

The project evaluates strategy performance under multiple transaction-cost assumptions:

```text
low_cost:    5 basis points per one-way trade
medium_cost: 10 basis points per one-way trade
high_cost:   25 basis points per one-way trade
```

The transaction-cost model calculates:

```text
turnover
gross return
transaction cost
net return
transaction-cost drag
```

## Portfolio Optimization

The risk-constrained optimization module is designed to support realistic allocation constraints, including:

* long-only weights
* maximum single-ETF weight
* minimum selected-ETF weight
* monthly turnover limits
* target volatility
* optional bond allocation limits
* optional commodity allocation limits

The objective is to balance expected return, portfolio risk, and turnover penalty.

A representative optimization objective is:

```text
maximize expected_return
- risk_aversion × portfolio_variance
- turnover_penalty × turnover
```

## Model-Risk Controls

The project includes explicit model-risk controls:

* no-lookahead feature validation
* chronological train/test splits
* feature-date checks
* transaction-cost stress scenarios
* benchmark comparison
* portfolio constraint checks
* reproducible outputs
* model-card documentation
* clear limitations and failure-case discussion

These controls are included to make the project credible as a research system rather than only a modeling exercise.

## Expected Results Files

### `results/selected_features.csv`

Contains the features selected for modeling, including information-theoretic relevance and redundancy information where available.

### `results/walk_forward_predictions.csv`

Contains out-of-sample predictions from the walk-forward validation process.

Expected columns include:

```text
month_end_date
ticker
actual_next_return
predicted_return
predicted_rank
model_name
feature_set
training_start
training_end
test_date
```

### `results/model_metrics.json`

Contains model-level validation metrics.

Typical metrics include:

```text
rank_ic
mean_absolute_error
root_mean_squared_error
directional_accuracy
hit_rate
```

### `results/feature_importance.csv`

Contains feature-importance values from the trained models.

### `results/portfolio_weights.csv`

Contains portfolio weights generated from model predictions and portfolio constraints.

### `results/backtest_results.csv`

Contains period-by-period backtest results, including gross return, transaction cost, and net return.

### `results/backtest_summary.json`

Contains summary performance metrics for the backtest.

### `reports/final_research_report.md`

Contains the full reproducible research report.

Expected sections include:

1. Executive Summary
2. Research Question
3. Data Sources
4. Hypothesis Registry
5. Feature Construction
6. GenAI/Text Feature Design
7. No-Lookahead Controls
8. Model Design
9. Walk-Forward Validation
10. Backtest Design
11. Transaction Cost Assumptions
12. Risk Constraints
13. Results
14. Failure Cases
15. Limitations
16. Future Work

### `reports/model_card.md`

Contains model-risk and governance documentation.

Expected sections include:

* model purpose
* data sources
* prediction target
* feature groups
* validation method
* known limitations
* model-risk controls
* ethical and practical limitations
* not-investment-advice disclaimer

## Setup

Create and activate a Python virtual environment.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies.

```powershell
pip install -r requirements.txt
```

## Running the Project

Run the full pipeline according to the available scripts in the repository.

Representative commands include:

```powershell
python src\models\train_baselines.py
python src\models\walk_forward_validation.py
python src\backtest\signal_to_portfolio.py
python src\backtest\transaction_costs.py
python src\backtest\portfolio_optimizer.py
python src\backtest\backtest_engine.py
python src\reporting\make_research_report.py
```

The exact sequence may depend on which data files have already been generated.

## Final Output Verification

Use this PowerShell check to confirm that the required final outputs exist:

```powershell
$required = @(
    "results\selected_features.csv",
    "results\walk_forward_predictions.csv",
    "results\model_metrics.json",
    "results\feature_importance.csv",
    "results\portfolio_weights.csv",
    "results\backtest_results.csv",
    "results\backtest_summary.json",
    "reports\final_research_report.md",
    "reports\model_card.md"
)

foreach ($file in $required) {
    if (Test-Path $file) {
        Write-Host "FOUND:   $file"
    } else {
        Write-Host "MISSING: $file"
    }
}
```

Expected result:

```text
FOUND:   results\selected_features.csv
FOUND:   results\walk_forward_predictions.csv
FOUND:   results\model_metrics.json
FOUND:   results\feature_importance.csv
FOUND:   results\portfolio_weights.csv
FOUND:   results\backtest_results.csv
FOUND:   results\backtest_summary.json
FOUND:   reports\final_research_report.md
FOUND:   reports\model_card.md
```

## Testing

Run the test suite:

```powershell
pytest
```

The tests are designed to check:

* no-lookahead behavior
* feature-date validity
* portfolio constraint validity
* backtest accounting
* transaction-cost logic
* walk-forward train/test ordering

## GitHub Commit Guidance

Recommended final commit:

```powershell
git add README.md results reports
git commit -m "Finalize FedNarrativeAlpha scope and research outputs"
git push
```

Do not commit large raw data files, local virtual environments, environment variables, or Python cache files.

Do not commit:

```text
data/raw/
data/interim/
.env
.venv/
__pycache__/
```

## Limitations

This project is a research demonstration and should not be interpreted as a live trading system.

Important limitations include:

* public data may be revised, delayed, or incomplete
* Federal Reserve text features may be noisy
* historical backtests may not generalize
* transaction costs and liquidity assumptions are simplified
* model performance may be unstable across regimes
* ETF relationships may change over time
* machine-learning models may overfit noisy market data

The project is designed to document these limitations explicitly rather than hide them.

## Disclaimer

This repository is for research, education, and portfolio demonstration purposes only.

It does not provide investment advice, trading recommendations, or financial planning guidance. Backtested results are hypothetical and do not guarantee future performance.

## Summary

FedNarrativeAlpha demonstrates a full quantitative research workflow:

```text
data
→ hypothesis
→ features
→ model
→ validation
→ portfolio construction
→ transaction-cost-aware backtest
→ risk analysis
→ reproducible report
→ model card
```

The project is strongest because it shows practical research engineering, time-series discipline, GenAI-assisted documentation, model validation, and clear communication of both results and limitations.
