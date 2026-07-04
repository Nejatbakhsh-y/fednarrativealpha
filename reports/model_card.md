# Model Card

**Project:** FedNarrativeAlpha: GenAI-Assisted Alpha Research Pipeline for Liquid ETF Markets
**Generated:** 2026-07-03 19:06:41
**Script:** `src/reporting/make_research_report.py`

---

## Model Purpose

The model predicts next-month relative ETF returns using price features, macroeconomic features, and Federal Reserve text-derived features. The research purpose is to test whether a GenAI-assisted narrative signal improves liquid ETF allocation decisions after transaction costs and risk constraints.

---

## Data Sources

| Artifact | Path | Status |
| --- | --- | --- |
| Price features | data/interim/price_features_monthly.parquet | Available |
| Macro features | data/interim/macro_features_monthly.parquet | Available |
| Fed text features | data/interim/fed_text_features_monthly.parquet | Available |
| Master modeling dataset | data/processed/master_modeling_dataset.parquet | Available |
| Selected features | results/selected_features.csv | Available |
| Baseline metrics | results/baseline_model_metrics.json | Missing |
| ML model metrics | results/model_metrics.json | Available |
| Walk-forward predictions | results/walk_forward_predictions.csv | Available |
| Raw signal weights | results/raw_signal_weights.csv | Available |
| Portfolio weights | results/portfolio_weights.csv | Available |
| Optimization diagnostics | results/optimization_diagnostics.csv | Available |
| Transaction-cost results | results/backtest_results.csv | Available |
| Backtest summary | results/backtest_summary.csv | Missing |

---

## Prediction Target

The main prediction target is `next_1m_return`.

Additional target fields are `next_1m_rank` and `top_tercile_next_month`.

---

## Feature Groups

1. Price and momentum features.
2. Realized volatility and drawdown features.
3. Volume and liquidity features.
4. Macroeconomic features.
5. Federal Reserve text and narrative features.
6. Information-theoretic selected features.

| Feature-selection attribute | Value |
| --- | --- |
| Rows | 12 |
| Columns | 7 |
| Selected features | 12 |
| Feature groups | 3 |

Top feature records:

| feature_name | feature_group | mutual_information | redundancy_penalty | final_score | selection_frequency | selected_flag |
| --- | --- | --- | --- | --- | --- | --- |
| realized_vol_21d | realized_volatility | 0.1795 | 0.0799 | 0.0996 | 1.0000 | True |
| realized_vol_63d | realized_volatility | 0.1492 | 0.0755 | 0.0737 | 1.0000 | True |
| realized_vol_126d | realized_volatility | 0.1373 | 0.0682 | 0.0691 | 1.0000 | True |
| return_1m | price_momentum | 0.0565 | 0.0466 | 0.0099 | 1.0000 | True |
| max_drawdown_6m | price_momentum | 0.0966 | 0.0967 | -0.0002 | 1.0000 | True |
| volume_change_3m | realized_volatility | 0.0006 | 0.0208 | -0.0202 | 1.0000 | True |
| momentum_12m_minus_1m | price_momentum | 0.0466 | 0.0682 | -0.0216 | 1.0000 | True |
| return_3m | price_momentum | 0.0551 | 0.0783 | -0.0231 | 1.0000 | True |
| dollar_volume | realized_volatility | 0.0079 | 0.0342 | -0.0262 | 1.0000 | True |
| return_12m | price_momentum | 0.0161 | 0.0795 | -0.0634 | 1.0000 | True |
| return_6m | price_momentum | 0.0256 | 0.0899 | -0.0643 | 1.0000 | True |
| price_above_200d_ma | other | 0.0181 | 0.0831 | -0.0650 | 1.0000 | True |

---

## Validation Method

The model uses walk-forward validation rather than random train-test splitting. The intended design uses a 60-month training window, a one-month test window, a one-month step size, and a five-trading-day embargo.

| Walk-forward attribute | Value |
| --- | --- |
| Rows | 43056 |
| Columns | 10 |
| Earliest test month | 2015-01-31 |
| Latest test month | 2026-06-30 |
| Unique tickers | 13 |
| Models | Elastic Net, Gradient Boosting, LightGBM, Random Forest, Ridge regression, XGBoost |
| Feature sets | Model A: price only, Model B: price + macro, Model C: price + macro + Fed text, Model D: selected features only |

Overall prediction diagnostics:

| Metric | Value |
| --- | --- |
| Usable prediction rows | 43056 |
| Prediction/actual correlation | 0.0030 |
| Sign hit rate | 0.5403 |

---

## Known Limitations

Known limitations include data revision risk, timestamp alignment risk, model instability, prompt sensitivity, market-regime dependence, and transaction-cost sensitivity.

---

## Model Risk Controls

1. Chronological walk-forward validation.
2. No-lookahead feature construction.
3. Embargo around validation periods.
4. Benchmark comparisons against non-text strategies.
5. Transaction-cost scenarios.
6. Turnover monitoring.
7. Risk-constrained portfolio optimization.
8. Feature-selection documentation.
9. Reproducible report generation.
10. Explicit model-card documentation.

---

## Ethical and Practical Limitations

This model is a research prototype. It is not designed to replace professional portfolio management, risk oversight, investment due diligence, or regulatory review.

---

## Not Investment Advice Disclaimer

This project is for educational and research purposes only. It does not provide investment advice, financial advice, trading recommendations, or an offer to buy or sell securities.
