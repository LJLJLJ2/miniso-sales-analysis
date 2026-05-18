# MINISO 2023 Sales Data Analysis

Comprehensive exploratory data analysis and sales prediction modeling of MINISO (名创优品) 2023 full-year sales data.

## Key Findings

- **Total Revenue**: ¥107.2M | **Total Units**: 905,530 | **9,125 records** (Jan-Dec 2023)
- **Most Effective Promotion**: Flash sales (限时特价) drive +74.4% sales lift vs no-promo baseline
- **Top Category**: Stationery & Office (文具办公) — 30.0% of total revenue
- **Counter-intuitive**: Weekend sales are 32% lower than weekdays
- **Holiday Boost**: Holidays drive +54.4% higher sales than non-holidays
- **Prediction Model**: Random Forest achieves R² = 0.66, MAE = ±16.2 units/day

## Project Structure

```
├── README.md
├── 分析报告.md                    # 方案一：EDA 分析报告 (Markdown)
├── 分析报告.html                  # 方案一：EDA 可视化报告
├── analysis.py                    # 方案一：EDA 分析脚本 (SQL + SciPy + Seaborn)
├── prediction_model.py            # 方案二：销量预测模型 (Random Forest + ARIMA)
├── prediction_report.html         # 方案二：预测模型可视化报告
├── prediction_metrics.json        # 方案二：模型性能指标
├── charts/                        # 方案一：13张分析图表
│   ├── 01_daily_trend.png
│   ├── 02_monthly.png
│   ├── ...
│   └── 13_violin_season.png
└── prediction_charts/             # 方案二：5张预测图表
    ├── feature_importance.png
    ├── prediction_vs_actual.png
    ├── residual_analysis.png
    ├── model_comparison.png
    ├── 30day_trend.png
    └── arima_detail.png
```

## Tools

**方案一 (EDA):** Python (pandas, matplotlib, seaborn) + SQL (SQLite) + SciPy (statistical tests)
**方案二 (Prediction):** Python (scikit-learn, statsmodels) — Random Forest + ARIMA

## Author

周琳杰 | Statistics & Economics background | AI-powered data analyst
