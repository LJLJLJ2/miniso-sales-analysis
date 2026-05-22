# MINISO 2023 Sales Data Analysis

> **在线报告：** [https://ljljlj2.github.io/miniso-sales-analysis/](https://ljljlj2.github.io/miniso-sales-analysis/)

Comprehensive data analysis portfolio: exploratory analysis → machine learning prediction → price elasticity & promotion ROI. MINISO (名创优品) full-year 2023 sales data.

## Key Findings

- **Total Revenue**: ¥107.2M | **Total Units**: 905,530 | **9,125 records** (Jan–Dec 2023)
- **Most Effective Promotion**: Flash sales (限时特价) drive +74.4% sales lift vs no-promo baseline
- **Best ROI Promotion**: Buy-one-get-one (买赠) ROI = 3.60 — 5× better than direct discount (打折, ROI = 0.72)
- **Price Elasticity ≈ 0**: All 24 products are price-inelastic — promotion type matters far more than exact price
- **Top Category**: Stationery & Office (文具办公) — 30.0% of total revenue
- **Counter-intuitive**: Weekend sales are 32% lower than weekdays; Holidays boost +54.4%
- **Best Model**: XGBoost R² = 0.67, MAE = ±16.0 units/day (after hyperparameter tuning)

## Three Analysis Modules

| Module | Description | Methods | Key Output |
|--------|-------------|---------|------------|
| **方案一** EDA | 8-dimension exploratory analysis | Pandas, Matplotlib, Seaborn, SciPy | 13 charts, HTML report |
| **方案二** Prediction | Sales forecasting with ML models | Random Forest, XGBoost, ARIMA | R²=0.67, 9 charts, tuning report |
| **方案三** Elasticity & ROI | Price elasticity + promotion ROI | Log-log regression, t-test, ROI model | 6 charts, 3 actionable recommendations |

## Project Structure

```
├── README.md
├── 分析报告.md                        # 方案一：EDA 分析报告 (Markdown)
├── 分析报告.html                      # 方案一：EDA 可视化报告
├── analysis.py                        # 方案一：EDA 分析脚本
│
├── prediction_model.py                # 方案二：销量预测 (RF + XGBoost + ARIMA)
├── tune_model.py                      # 方案二+：超参数调优脚本
├── prediction_report.html             # 方案二：预测模型可视化报告
├── prediction_metrics.json            # 方案二：模型性能指标
├── tuning_result.json                 # 方案二+：调优结果
│
├── price_elasticity_analysis.py       # 方案三：价格弹性与促销ROI分析
├── elasticity_report.html             # 方案三：弹性分析可视化报告
├── elasticity_metrics.json            # 方案三：核心指标
│
├── charts/                            # 方案一：13张EDA图表
├── prediction_charts/                 # 方案二：9张预测图表（含调优对比）
└── elasticity_charts/                 # 方案三：6张弹性与ROI图表
```

## Key Charts

### 方案一 EDA — 促销效果对比
![Promotion Effect](https://raw.githubusercontent.com/LJLJLJ2/miniso-sales-analysis/main/charts/05_promotion.png)

*限时特价效果最强（+74.4%），买赠次之（+58.3%），四种促销方式均有显著正向效果（P<0.0001）*

### 方案二 Prediction — 模型对比
![Model Comparison](https://raw.githubusercontent.com/LJLJLJ2/miniso-sales-analysis/main/prediction_charts/model_comparison.png)

*XGBoost (R²=0.67) 优于 Random Forest (R²=0.61) 和 ARIMA (R²=0.36)，调优后 MAE ±16 件/天*

### 方案三 Elasticity — 促销 ROI 排名
![Promotion ROI](https://raw.githubusercontent.com/LJLJLJ2/miniso-sales-analysis/main/elasticity_charts/promotion_roi.png)

*买赠 ROI 3.60 > 满减 2.21 > 限时特价 1.84 > 打折 0.72，打折实际赔钱*

## Tools & Methods

**方案一 (EDA):** Python (pandas, matplotlib, seaborn) + SciPy (statistical tests)

**方案二 (Prediction):** Python (scikit-learn, XGBoost, statsmodels) — Random Forest + XGBoost + ARIMA + RandomizedSearchCV hyperparameter tuning

**方案三 (Elasticity & ROI):** Python (scikit-learn, SciPy) — Log-log OLS regression + independent t-tests + ROI estimation model

## About

Built as a data analysis portfolio project showcasing the full pipeline: **Describe → Predict → Prescribe**.

Author: 周琳杰 | Statistics & Economics background
