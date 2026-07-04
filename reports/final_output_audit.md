# Final Output Audit

## Required Output Files

| file                                            | exists   | non_empty   |   size_bytes | status   |
|:------------------------------------------------|:---------|:------------|-------------:|:---------|
| data/processed/master_modeling_dataset.parquet  | True     | True        |       341037 | OK       |
| results/selected_features.csv                   | True     | True        |         1349 | OK       |
| results/walk_forward_predictions.csv            | True     | True        |      5763489 | OK       |
| results/model_metrics.json                      | True     | True        |        71398 | OK       |
| results/feature_importance.csv                  | True     | True        |       265991 | OK       |
| results/portfolio_weights.csv                   | True     | True        |       118982 | OK       |
| results/backtest_results.csv                    | True     | True        |        67487 | OK       |
| results/backtest_summary.json                   | True     | True        |          567 | OK       |
| reports/final_research_report.md                | True     | True        |        71456 | OK       |
| reports/model_card.md                           | True     | True        |         5247 | OK       |
| reports/figures/cumulative_returns.png          | True     | True        |        43792 | OK       |
| reports/figures/drawdown_curve.png              | True     | True        |        53125 | OK       |
| reports/figures/feature_importance.png          | True     | True        |        36296 | OK       |
| reports/figures/walk_forward_rank_ic.png        | True     | True        |       174518 | OK       |
| reports/figures/portfolio_weights_over_time.png | True     | True        |       600395 | OK       |

## Validation Result

Status: PASSED

All required final GitHub outputs exist and passed basic validation checks.