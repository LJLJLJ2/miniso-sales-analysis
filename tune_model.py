#!/usr/bin/env python3
"""超参数调优 —— Random Forest + XGBoost，搜最优参数组合"""

import pandas as pd
import numpy as np
import json, warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import xgboost as xgb

sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['Heiti TC', 'STHeiti', 'Hiragino Sans GB', 'Arial Unicode MS']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

DATA = '/Users/zhoujingjing/miniso_sales_data_features.csv'
OUT = '/Users/zhoujingjing/Desktop/miniso_sales_analysis'
C5 = ['#1b4b5e','#e8733a','#4ca6ba','#c39858','#8e6c8a']

# ═══════════════════════════════════
# 1. 加载数据（和原脚本完全一样）
# ═══════════════════════════════════
print("1. 加载数据...")
df = pd.read_csv(DATA, parse_dates=['date'])
df = df.sort_values(['product', 'date']).reset_index(drop=True)

LEAK_COLS = ['sales_amount', 'inventory', 'turnover_rate']
NUMERIC_FEATURES = [
    'base_price', 'month', 'weekday', 'is_holiday',
    'day_of_year', 'week_of_year', 'quarter',
    'sales_lag_1', 'sales_lag_7', 'sales_lag_30',
    'sales_7d_mean', 'sales_7d_std', 'sales_30d_mean',
]
CATEGORICAL_FEATURES = ['category', 'product', 'price_level', 'promotion_type', 'weather', 'season']

for col in CATEGORICAL_FEATURES:
    le = LabelEncoder()
    df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))

FEATURE_COLS = [c for c in (
    NUMERIC_FEATURES
    + [f'{col}_encoded' for col in CATEGORICAL_FEATURES]
    + ['price_level_encoded', 'weather_encoded', 'season_encoded', 'promotion_encoded',
       'price_promotion_interaction', 'season_weather_interaction']
) if c in df.columns and c not in LEAK_COLS]
FEATURE_COLS = list(dict.fromkeys(FEATURE_COLS))

lag_cols = [c for c in FEATURE_COLS if 'lag' in c or 'mean' in c or 'std' in c]
for col in lag_cols:
    if col in df.columns:
        df[col] = df.groupby('product')[col].transform(lambda x: x.fillna(x.mean()))
        df[col] = df[col].fillna(df[col].mean())

TARGET = 'sales'

# 时间切分：训练=1~11月，测试=12月
split_date = pd.Timestamp('2023-12-01')
train_mask = df['date'] < split_date
test_mask = df['date'] >= split_date

X_train, y_train = df.loc[train_mask, FEATURE_COLS], df.loc[train_mask, TARGET]
X_test, y_test = df.loc[test_mask, FEATURE_COLS], df.loc[test_mask, TARGET]

print(f"   训练集: {len(X_train)} 条, 测试集: {len(X_test)} 条, 特征: {len(FEATURE_COLS)} 个\n")

# ═══════════════════════════════════
# 2. 定义搜索空间
# ═══════════════════════════════════
# TimeSeriesSplit: 按时间顺序切分，不用随机打乱
tscv = TimeSeriesSplit(n_splits=5)

rf_params = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [5, 8, 10, 15, 20, 25, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
}

xgb_params = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5],
}

# ═══════════════════════════════════
# 3. Random Forest 调优
# ═══════════════════════════════════
print("2. Random Forest 超参数搜索（50 组随机组合 × 5 折时间序列交叉验证）...")
rf_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    rf_params,
    n_iter=50,
    cv=tscv,
    scoring='neg_mean_squared_error',
    random_state=42,
    n_jobs=-1,
    verbose=1,
)
rf_search.fit(X_train, y_train)
print(f"   ✅ 最优参数: {rf_search.best_params_}")
print(f"   ✅ 最优 CV 得分 (neg_MSE): {rf_search.best_score_:.2f}")

# ═══════════════════════════════════
# 4. XGBoost 调优
# ═══════════════════════════════════
print("\n3. XGBoost 超参数搜索（50 组随机组合 × 5 折时间序列交叉验证）...")
xgb_search = RandomizedSearchCV(
    xgb.XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
    xgb_params,
    n_iter=50,
    cv=tscv,
    scoring='neg_mean_squared_error',
    random_state=42,
    n_jobs=-1,
    verbose=1,
)
xgb_search.fit(X_train, y_train)
print(f"   ✅ 最优参数: {xgb_search.best_params_}")
print(f"   ✅ 最优 CV 得分 (neg_MSE): {xgb_search.best_score_:.2f}")

# ═══════════════════════════════════
# 5. 用最优参数在测试集上评估
# ═══════════════════════════════════
print("\n4. 测试集评估 —— 调优前 vs 调优后")

# 原参数（调优前）
rf_old = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf_old.fit(X_train, y_train)
y_pred_rf_old = rf_old.predict(X_test)

xgb_old = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbosity=0)
xgb_old.fit(X_train, y_train)
y_pred_xgb_old = xgb_old.predict(X_test)

# 最优参数（调优后）
rf_new = rf_search.best_estimator_
y_pred_rf_new = rf_new.predict(X_test)

xgb_new = xgb_search.best_estimator_
y_pred_xgb_new = xgb_new.predict(X_test)

# ── 汇总对比 ──
def metrics_dict(y_true, y_pred, name):
    return {
        '模型': name,
        'MAE': round(mean_absolute_error(y_true, y_pred), 1),
        'RMSE': round(np.sqrt(mean_squared_error(y_true, y_pred)), 1),
        'R²': round(r2_score(y_true, y_pred), 4),
        'MAPE': round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100, 1),
    }

results = [
    metrics_dict(y_test, y_pred_rf_old, 'RF 调优前 (100树/15层)'),
    metrics_dict(y_test, y_pred_rf_new, f'RF 调优后 ({rf_search.best_params_["n_estimators"]}树/{rf_search.best_params_["max_depth"]}层)'),
    metrics_dict(y_test, y_pred_xgb_old, 'XGBoost 调优前 (100轮/6层/0.1率)'),
    metrics_dict(y_test, y_pred_xgb_new, f'XGBoost 调优后 ({xgb_search.best_params_["n_estimators"]}轮/{xgb_search.best_params_["max_depth"]}层/{xgb_search.best_params_["learning_rate"]}率)'),
]

print(f"\n{'模型':<45} {'MAE':>7} {'RMSE':>7} {'R²':>8} {'MAPE':>8}")
print("-" * 78)
for r in results:
    print(f"{r['模型']:<45} {r['MAE']:>7.1f} {r['RMSE']:>7.1f} {r['R²']:>8.4f} {r['MAPE']:>7.1f}%")

# 算提升
r2_improve_rf = results[1]['R²'] - results[0]['R²']
r2_improve_xgb = results[3]['R²'] - results[2]['R²']
mae_improve_rf = (results[0]['MAE'] - results[1]['MAE']) / results[0]['MAE'] * 100
mae_improve_xgb = (results[2]['MAE'] - results[3]['MAE']) / results[2]['MAE'] * 100

print(f"\n   RF  R² 提升: +{r2_improve_rf:.4f}  |  MAE 降低: {mae_improve_rf:.1f}%")
print(f"   XGB R² 提升: +{r2_improve_xgb:.4f}  |  MAE 降低: {mae_improve_xgb:.1f}%")

# ═══════════════════════════════════
# 6. 可视化对比
# ═══════════════════════════════════
print("\n5. 生成对比图表...")

# Chart 1: 调优前后 R² 对比柱状图
fig, ax = plt.subplots(figsize=(10, 5.5))
models_label = ['RF\n调优前', 'RF\n调优后', 'XGBoost\n调优前', 'XGBoost\n调优后']
r2_vals = [results[0]['R²'], results[1]['R²'], results[2]['R²'], results[3]['R²']]
colors = [C5[0], C5[0], C5[1], C5[1]]
alphas = [0.5, 1.0, 0.5, 1.0]
bars = ax.bar(models_label, r2_vals, color=colors, width=0.55)
for bar, a in zip(bars, alphas):
    bar.set_alpha(a)
for bar, val in zip(bars, r2_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f'{val:.4f}', ha='center', fontsize=12, fontweight='bold')
ax.set_ylabel('R² (越高越好)', fontsize=12)
ax.set_title('超参数调优前后 R² 对比', fontsize=14, fontweight='bold')
ax.set_ylim(0, max(r2_vals)*1.15)
ax.axhline(y=results[0]['R²'], color=C5[0], linewidth=0.8, linestyle=':', alpha=0.4)
plt.tight_layout()
fig.savefig(f'{OUT}/prediction_charts/tuning_r2_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 2: 调优后预测 vs 实际（最佳模型）
best_idx = max(range(4), key=lambda i: results[i]['R²'])
best_model_name = results[best_idx]['模型']
if best_idx <= 1:
    y_pred_best = y_pred_rf_new if best_idx == 1 else y_pred_rf_old
else:
    y_pred_best = y_pred_xgb_new if best_idx == 3 else y_pred_xgb_old

fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(y_test, y_pred_best, alpha=0.3, s=15, color=C5[0], edgecolors='none')
lim_min = min(y_test.min(), y_pred_best.min()) - 10
lim_max = max(y_test.max(), y_pred_best.max()) + 10
ax.plot([lim_min, lim_max], [lim_min, lim_max], '--', color=C5[1], linewidth=2, label='完美预测线')
ax.set_xlim(lim_min, lim_max); ax.set_ylim(lim_min, lim_max)
ax.set_xlabel('实际销量（件）', fontsize=12)
ax.set_ylabel('预测销量（件）', fontsize=12)
ax.set_title(f'最佳模型: {best_model_name}\nR²={results[best_idx]["R²"]:.3f}, MAE={results[best_idx]["MAE"]:.1f}件', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.set_aspect('equal')
plt.tight_layout()
fig.savefig(f'{OUT}/prediction_charts/tuning_best_model.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 3: 调优前后每日预测趋势对比
fig, axes = plt.subplots(2, 1, figsize=(14, 9))
df_test = df.loc[test_mask, ['date', 'sales']].copy()
daily_actual = df_test.groupby('date')['sales'].sum()

for i, (name_old, pred_old, name_new, pred_new, color) in enumerate([
    ('RF 调优前', y_pred_rf_old, 'RF 调优后', y_pred_rf_new, C5[0]),
    ('XGBoost 调优前', y_pred_xgb_old, 'XGBoost 调优后', y_pred_xgb_new, C5[1]),
]):
    ax = axes[i]
    ax.plot(daily_actual.index, daily_actual.values, 'o-', color='#333', linewidth=2, markersize=5, label='实际销量')
    daily_old = pd.Series(pred_old, index=df_test.index).groupby(df_test['date']).sum()
    daily_new = pd.Series(pred_new, index=df_test.index).groupby(df_test['date']).sum()
    ax.plot(daily_old.index, daily_old.values, 's--', color=color, alpha=0.45, linewidth=1.8, markersize=4, label=name_old)
    ax.plot(daily_new.index, daily_new.values, 's-', color=color, alpha=1.0, linewidth=2.2, markersize=5, label=name_new)
    ax.set_title(f'{name_old.split()[0]} —— 调优前后每日总销量预测对比', fontsize=13, fontweight='bold')
    ax.set_ylabel('总销量（件）', fontsize=11)
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m-%d'))
    ax.xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=3))
fig.autofmt_xdate()
plt.tight_layout()
fig.savefig(f'{OUT}/prediction_charts/tuning_daily_trend.png', dpi=150, bbox_inches='tight')
plt.close()

print('   ✅ 对比图表已生成')

# ═══════════════════════════════════
# 7. 保存结果
# ═══════════════════════════════════
tuning_result = {
    '调优前': {
        'RandomForest': results[0],
        'XGBoost': results[2],
    },
    '调优后': {
        'RandomForest': {**results[1], 'best_params': rf_search.best_params_},
        'XGBoost': {**results[3], 'best_params': xgb_search.best_params_},
    },
    'R²提升_RF': round(r2_improve_rf, 4),
    'R²提升_XGB': round(r2_improve_xgb, 4),
    'MAE降低%_RF': round(mae_improve_rf, 1),
    'MAE降低%_XGB': round(mae_improve_xgb, 1),
}

with open(f'{OUT}/tuning_result.json', 'w') as f:
    json.dump(tuning_result, f, ensure_ascii=False, indent=2)

print(f"\n6. 调优结果已保存到 tuning_result.json")
print("✅ 全部完成！")
