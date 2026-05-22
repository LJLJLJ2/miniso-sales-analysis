# MINISO 2023 Sales Data Analysis

> **在线报告：** [https://ljljlj2.github.io/miniso-sales-analysis/](https://ljljlj2.github.io/miniso-sales-analysis/)

Comprehensive data analysis portfolio: exploratory analysis → machine learning prediction → price elasticity & promotion ROI. MINISO (名创优品) full-year 2023 sales data.

## Data Source

Data from [Kaggle — MINISO Sales Dataset](https://www.kaggle.com/) (名创优品 2023 全年销售明细). The dataset contains **9,125 daily records** across **24 products** in **8 categories** from Jan 1 to Dec 31, 2023, with fields including price, sales volume, inventory, promotion type, weather, and holidays.

> **Note:** The raw CSV is not included in this repo (Kaggle dataset license). Download it from Kaggle and run `python feature_engineering.py` to reproduce the full pipeline. See [Data Pipeline](#data-pipeline) below.

## Data Pipeline

```
Raw CSV (Kaggle)  →  feature_engineering.py  →  features CSV  →  方案一 / 方案二 / 方案三
     9,125 rows             15 new features         31 columns      EDA / ML / ROI
```

`feature_engineering.py` transforms raw transaction data into analysis-ready features. Each step is motivated by a specific business or modeling need:

| Step | Features Created | Business / Modeling Rationale |
|------|-----------------|------------------------------|
| **Temporal extraction** | `day_of_year`, `week_of_year`, `quarter` | 零售销量有强周期性和季节性——季度驱动财报周期，周数捕捉周级别趋势 |
| **Lag features** | `sales_lag_1`, `lag_7`, `lag_30` | 销量预测核心假设：「过去预示未来」——日惯性、周模式、月趋势三层捕捉 |
| **Rolling statistics** | `sales_7d_mean`, `7d_std`, `30d_mean` | 滚动均值反映短期/中期趋势，标准差告诉模型该产品波动性大小 |
| **Categorical encoding** | `price_level_enc`, `weather_enc`, `season_enc`, `promotion_enc` | 树模型需要数值输入；Label Encoding 避免 One-Hot 对 24 产品的维度膨胀 |
| **Interaction terms** | `price×promotion`, `season×weather` | 同一促销在不同价格带效果不同；同一季节不同天气影响不同——组合效应需显式建模 |
| **NaN handling** | (在 prediction_model.py 中处理) | Lag 特征的产品早期记录为 NaN，按产品组均值填充，避免信息泄露 |

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
| **方案二** Prediction | Sales forecasting — inventory & staffing planning | Random Forest, XGBoost, ARIMA | R²=0.67, MAE±16件: usable for weekly planning |
| **方案三** Elasticity & ROI | Price elasticity + promotion ROI | Log-log regression, t-test, ROI model | 6 charts, 3 actionable recommendations |

## Project Structure

```
├── README.md
├── requirements.txt
├── feature_engineering.py             # 数据流水线：原始CSV → 特征工程 → 分析就绪
│
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

*XGBoost (R²=0.67) 优于 Random Forest (R²=0.61) 和 ARIMA (R²=0.36)。ARIMA 仅捕获时序模式、R² 最低——说明销量不由季节主导，**由促销、定价等运营动作驱动**，这正是树模型的优势。日均销量 ~100 件水平下，MAE ±16 件（~16% 误差）可支撑周度备货计划和人力排班，不宜用于日级精确补货。*

![Feature Importance](https://raw.githubusercontent.com/LJLJLJ2/miniso-sales-analysis/main/prediction_charts/feature_importance.png)

*特征重要性 Top 5：短期滞后销量、促销类型、价格带、周中/周末、滚动均值——**运营类特征压倒时间类特征**，验证了「零售销量是管理出来的，不是等出来的」*

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
