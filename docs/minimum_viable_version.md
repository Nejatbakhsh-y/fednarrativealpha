# Minimum Viable Version Roadmap

This project is built in two stages.

The first stage is the minimum viable quant pipeline. The purpose is to prove that the repository can ingest market and macroeconomic data, construct no-lookahead monthly features, train models with chronological validation, convert predictions into portfolio weights, apply transaction costs, and generate reproducible research outputs.

The second stage adds the GenAI/text component. The text layer should only be added after the baseline quant pipeline is complete and verified.

## Stage 1: Minimum Strong Quant Pipeline

The minimum viable version must complete the following steps first:

1. Download ETF price data.
2. Download FRED macroeconomic data.
3. Build monthly price and macro features.
4. Create the next-month ETF return target.
5. Train baseline models, Random Forest models, and XGBoost models.
6. Use walk-forward validation rather than random train-test splits.
7. Backtest a top-3 ETF strategy based on predicted returns.
8. Add transaction costs.
9. Produce the required model and report outputs.

The minimum required outputs are:

```text
data/processed/master_modeling_dataset.parquet
results/model_metrics.json
results/walk_forward_predictions.csv
results/backtest_results.csv
reports/final_research_report.md
```

This stage is the credibility foundation of the project. It shows that the project can run from data ingestion through validation, portfolio construction, backtesting, and reporting.

## Stage 2: GenAI/Text Layer

After the minimum quant pipeline is working, the project adds the Federal Reserve narrative layer:

10. Scrape FOMC statements.
11. Create sentence-transformer embeddings.
12. Add Fed narrative features to the monthly modeling dataset.
13. Compare model performance with and without text features.
14. Generate the model card.

The final comparison should answer the central research question:

Does adding the Fed narrative feature layer improve out-of-sample ETF return prediction and portfolio performance after transaction costs?

## Required Model Comparison

The final project should compare at least these feature groups:

```text
Model A: Price features only
Model B: Price + macro features
Model C: Price + macro + Fed text features
Model D: Selected features only
```

The main success criterion is not in-sample accuracy. The main criterion is whether the full model improves walk-forward validation results and backtested portfolio performance after transaction costs.

## Why This Sequencing Matters

The project should not start with embeddings, language models, or narrative features. Those components only matter if the basic quant pipeline is already correct.

The correct sequence is:

```text
data -> features -> target -> validation -> prediction -> portfolio -> transaction costs -> report -> text features -> model card
```

This prevents the project from becoming an unfocused AI demo. The minimum viable version establishes financial modeling discipline first. The GenAI layer is then added as a measurable research extension.

## Completion Criteria

Step 25 is complete when the repository documents this two-stage build sequence and the minimum strong version can produce the core files needed for the final project.

The project should be considered incomplete if it has Fed text features but lacks any of the following:

```text
walk-forward predictions
transaction-cost-adjusted returns
model metrics
final research report
model card
```

The project should be considered credible if it can produce the full quant pipeline first, then show whether the Fed text layer improves results.
