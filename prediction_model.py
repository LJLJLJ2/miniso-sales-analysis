#!/usr/bin/env python3
"""MINISO 2023 Sales Prediction — Random Forest + XGBoost + ARIMA"""

import pandas as pd
import numpy as np
import os, sys, warnings, json
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# ── ML ──
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# ── XGBoost (needs libomp — bundled with sklearn) ──
import os as _os
_os.environ.setdefault('DYLD_LIBRARY_PATH', _os.path.expanduser('~/lib'))
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False
    print("⚠ xgboost not available (libomp missing), skipping XGBoost model")

# ── Statsmodels (ARIMA) ──
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    HAS_ARIMA = True
except ImportError:
    HAS_ARIMA = False
    print("⚠ statsmodels not installed, skipping ARIMA model")

# ── Setup ──
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['Heiti TC', 'STHeiti', 'Hiragino Sans GB', 'Arial Unicode MS']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

DATA_FEATURES = '/Users/zhoujingjing/miniso_sales_data_features.csv'
DATA_RAW = '/Users/zhoujingjing/miniso_sales_data.csv'
OUT = '/Users/zhoujingjing/Desktop/miniso_sales_analysis'
os.makedirs(f'{OUT}/prediction_charts', exist_ok=True)

C5 = ['#1b4b5e','#e8733a','#4ca6ba','#c39858','#8e6c8a']

# ═══════════════════════════════════════════
# 1. LOAD & PREPARE DATA
# ═══════════════════════════════════════════
print("=" * 60)
print("1. LOADING DATA")
print("=" * 60)

df = pd.read_csv(DATA_FEATURES, parse_dates=['date'])
df = df.sort_values(['product', 'date']).reset_index(drop=True)

print(f"Shape: {df.shape}")
print(f"Date range: {df['date'].min().date()} ~ {df['date'].max().date()}")

# ── 1a. Identify data leak columns ──
LEAK_COLS = ['sales_amount', 'inventory', 'turnover_rate']
print(f"\n🔒 Excluding DATA LEAK columns: {LEAK_COLS}")
print("   These columns contain information that wouldn't be known at prediction time.")

# ── 1b. Define feature set ──
NUMERIC_FEATURES = [
    'base_price', 'month', 'weekday', 'is_holiday',
    'day_of_year', 'week_of_year', 'quarter',
    'sales_lag_1', 'sales_lag_7', 'sales_lag_30',
    'sales_7d_mean', 'sales_7d_std', 'sales_30d_mean',
]
CATEGORICAL_FEATURES = [
    'category', 'product', 'price_level', 'promotion_type', 'weather', 'season'
]
ENCODED_FEATURES = [
    'price_level_encoded', 'weather_encoded', 'season_encoded', 'promotion_encoded',
    'price_promotion_interaction', 'season_weather_interaction'
]

# ── 1c. Encode categorical columns ──
print("\nEncoding categorical features...")
encoders = {}
for col in CATEGORICAL_FEATURES:
    le = LabelEncoder()
    df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Build final feature list
FEATURE_COLS = (NUMERIC_FEATURES
                + [f'{col}_encoded' for col in CATEGORICAL_FEATURES]
                + ENCODED_FEATURES)

# Remove duplicate columns
FEATURE_COLS = list(dict.fromkeys(FEATURE_COLS))
# Keep only columns that exist in df
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns and c not in LEAK_COLS]

TARGET = 'sales'

print(f"Features: {len(FEATURE_COLS)} columns")
print(f"Target: {TARGET}")

# ── 1d. Handle NaN in lag features ──
# Lag features have NaN at the start of each product sequence
# Fill with the product's overall mean sales
print("\nFilling NaN values in lag features...")
lag_cols = [c for c in FEATURE_COLS if 'lag' in c or 'mean' in c or 'std' in c]
for col in lag_cols:
    if col in df.columns:
        df[col] = df.groupby('product')[col].transform(lambda x: x.fillna(x.mean()))
        df[col] = df[col].fillna(df[col].mean())  # fallback

# Verify no NaN remains
nan_count = df[FEATURE_COLS].isna().sum().sum()
print(f"Remaining NaN in features: {nan_count}")

# ═══════════════════════════════════════════
# 2. TRAIN / TEST SPLIT (Time-based!)
# ═══════════════════════════════════════════
print("\n" + "=" * 60)
print("2. TRAIN / TEST SPLIT (Time-based)")
print("=" * 60)

split_date = pd.Timestamp('2023-12-01')
train_mask = df['date'] < split_date
test_mask = df['date'] >= split_date

X_train, y_train = df.loc[train_mask, FEATURE_COLS], df.loc[train_mask, TARGET]
X_test, y_test = df.loc[test_mask, FEATURE_COLS], df.loc[test_mask, TARGET]

# Also keep test metadata for visualization
df_test_meta = df.loc[test_mask, ['date', 'product', 'category', 'sales']].copy()

print(f"Train: {len(X_train)} rows (2023-01-01 ~ 2023-11-30)")
print(f"Test:  {len(X_test)} rows (2023-12-01 ~ 2023-12-31)")
print(f"Train/test ratio: {len(X_train)/len(df)*100:.1f}% / {len(X_test)/len(df)*100:.1f}%")

# ═══════════════════════════════════════════
# 3. MODEL A: RANDOM FOREST
# ═══════════════════════════════════════════
print("\n" + "=" * 60)
print("3. MODEL A: RANDOM FOREST")
print("=" * 60)

print("Training Random Forest (100 trees)...")
rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)
mape_rf = np.mean(np.abs((y_test - y_pred_rf) / y_test)) * 100

print(f"  MAE  = {mae_rf:.1f} 件  (平均每次预测差 ±{mae_rf:.0f} 件)")
print(f"  RMSE = {rmse_rf:.1f} 件  (大误差被放大惩罚)")
print(f"  R²   = {r2_rf:.4f}  ({'✅ 好' if r2_rf > 0.7 else '⚠ 一般' if r2_rf > 0.4 else '❌ 差'})")
print(f"  MAPE = {mape_rf:.1f}%  (平均误差百分比)")

# Feature importance
rf_importance = pd.DataFrame({
    'feature': FEATURE_COLS,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 important features (Random Forest):")
for _, row in rf_importance.head(10).iterrows():
    print(f"  {row['feature']:30s}  {row['importance']:.4f}")

# ═══════════════════════════════════════════
# 4. MODEL B: XGBOOST
# ═══════════════════════════════════════════
if HAS_XGB:
    print("\n" + "=" * 60)
    print("4. MODEL B: XGBoost")
    print("=" * 60)

    print("Training XGBoost (100 rounds)...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=42, n_jobs=-1, verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)

    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    r2_xgb = r2_score(y_test, y_pred_xgb)
    mape_xgb = np.mean(np.abs((y_test - y_pred_xgb) / y_test)) * 100

    print(f"  MAE  = {mae_xgb:.1f} 件")
    print(f"  RMSE = {rmse_xgb:.1f} 件")
    print(f"  R²   = {r2_xgb:.4f}  ({'✅ 好' if r2_xgb > 0.7 else '⚠ 一般' if r2_xgb > 0.4 else '❌ 差'})")
    print(f"  MAPE = {mape_xgb:.1f}%")

    xgb_importance = pd.DataFrame({
        'feature': FEATURE_COLS,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 10 important features (XGBoost):")
    for _, row in xgb_importance.head(10).iterrows():
        print(f"  {row['feature']:30s}  {row['importance']:.4f}")
else:
    mae_xgb = rmse_xgb = r2_xgb = mape_xgb = None
    y_pred_xgb = None


# ═══════════════════════════════════════════
# 5. MODEL C: ARIMA (per top product)
# ═══════════════════════════════════════════
arima_results = []
if HAS_ARIMA:
    print("\n" + "=" * 60)
    print("5. MODEL C: ARIMA (per product, top 5)")
    print("=" * 60)

    # Get top 5 products by total sales
    top5_products = df.groupby('product')['sales'].sum().nlargest(5).index.tolist()
    # Make sure they're strings
    top5_products = [str(p) for p in top5_products]

    print(f"Training ARIMA for top 5 products: {top5_products}")

    for prod_name in top5_products:
        # Get this product's daily sales (aggregate across categories if needed)
        prod_mask = df['product'] == prod_name
        prod_daily = df[prod_mask].groupby('date')['sales'].sum().reset_index()
        prod_daily = prod_daily.sort_values('date')

        # Split train/test
        prod_train = prod_daily[prod_daily['date'] < split_date]['sales'].values
        prod_test = prod_daily[prod_daily['date'] >= split_date]['sales'].values

        if len(prod_train) < 30 or len(prod_test) < 5:
            print(f"  {prod_name}: insufficient data, skipping")
            continue

        try:
            # Simple ARIMA(2,0,2) - fast and often good enough
            model = ARIMA(prod_train, order=(2, 0, 2))
            fitted = model.fit()
            preds = fitted.forecast(steps=len(prod_test))

            prod_mae = mean_absolute_error(prod_test, preds)
            prod_rmse = np.sqrt(mean_squared_error(prod_test, preds))

            arima_results.append({
                'product': prod_name,
                'MAE': prod_mae,
                'RMSE': prod_rmse,
                'preds': preds,
                'actual': prod_test,
                'dates': prod_daily[prod_daily['date'] >= split_date]['date'].values
            })
            print(f"  {prod_name}: MAE={prod_mae:.1f}, RMSE={prod_rmse:.1f}")

        except Exception as e:
            print(f"  {prod_name}: ARIMA failed - {str(e)[:60]}")

    if arima_results:
        avg_mae_arima = np.mean([r['MAE'] for r in arima_results])
        avg_rmse_arima = np.mean([r['RMSE'] for r in arima_results])
        print(f"\nARIMA avg (top 5): MAE={avg_mae_arima:.1f}, RMSE={avg_rmse_arima:.1f}")
    else:
        avg_mae_arima = avg_rmse_arima = None
        print("\nARIMA: no successful models")
else:
    avg_mae_arima = avg_rmse_arima = None

# ═══════════════════════════════════════════
# 6. VISUALIZATIONS
# ═══════════════════════════════════════════
print("\n" + "=" * 60)
print("6. VISUALIZATIONS")
print("=" * 60)

# ── Chart 1: Feature Importance (RF) ──
fig, ax = plt.subplots(figsize=(10, 5.5))
top15 = rf_importance.head(15).sort_values('importance', ascending=True)
bars = ax.barh(top15['feature'], top15['importance'], color=C5[0], height=0.6)
for bar, val in zip(bars, top15['importance']):
    ax.text(bar.get_width()+0.003, bar.get_y()+bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9)
ax.set_title('Feature Importance — Random Forest', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance (越高越重要)', fontsize=11)
plt.tight_layout()
fig.savefig(f'{OUT}/prediction_charts/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print('Chart: Feature Importance done')

# ── Chart 2: Prediction vs Actual scatter ──
fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(y_test, y_pred_rf, alpha=0.3, s=15, color=C5[0], edgecolors='none')
lim_min = min(y_test.min(), y_pred_rf.min()) - 10
lim_max = max(y_test.max(), y_pred_rf.max()) + 10
ax.plot([lim_min, lim_max], [lim_min, lim_max], '--', color=C5[1], linewidth=2, label='完美预测线')
ax.set_xlim(lim_min, lim_max); ax.set_ylim(lim_min, lim_max)
ax.set_xlabel('实际销量（件）', fontsize=12)
ax.set_ylabel('预测销量（件）', fontsize=12)
ax.set_title(f'Random Forest: 预测 vs 实际\nR²={r2_rf:.3f}, MAE={mae_rf:.1f}件', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.set_aspect('equal')
plt.tight_layout()
fig.savefig(f'{OUT}/prediction_charts/prediction_vs_actual.png', dpi=150, bbox_inches='tight')
plt.close()
print('Chart: Prediction vs Actual done')

# ── Chart 3: Residual Analysis ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
residuals_rf = y_test - y_pred_rf
ax1.hist(residuals_rf, bins=50, color=C5[0], alpha=0.8, edgecolor='white')
ax1.axvline(0, color=C5[1], linewidth=2, linestyle='--')
ax1.set_title('残差分布 (Random Forest)', fontsize=13, fontweight='bold')
ax1.set_xlabel('预测误差（实际−预测）', fontsize=11); ax1.set_ylabel('频次', fontsize=11)

ax2.scatter(y_pred_rf, residuals_rf, alpha=0.3, s=15, color=C5[0], edgecolors='none')
ax2.axhline(0, color=C5[1], linewidth=2, linestyle='--')
ax2.set_title('残差 vs 预测值', fontsize=13, fontweight='bold')
ax2.set_xlabel('预测销量（件）', fontsize=11); ax2.set_ylabel('残差（件）', fontsize=11)
plt.tight_layout()
fig.savefig(f'{OUT}/prediction_charts/residual_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print('Chart: Residual Analysis done')

# ── Chart 4: Model Comparison ──
fig, ax = plt.subplots(figsize=(10, 5))
models = ['Random Forest']
maes = [mae_rf]
rmses = [rmse_rf]
r2s = [r2_rf]
colors = [C5[0]]

if HAS_XGB:
    models.append('XGBoost')
    maes.append(mae_xgb)
    rmses.append(rmse_xgb)
    r2s.append(r2_xgb)
    colors.append(C5[1])

if arima_results and avg_mae_arima is not None:
    models.append('ARIMA\n(avg top5)')
    maes.append(avg_mae_arima)
    rmses.append(avg_rmse_arima)
    colors.append(C5[2])
    # R² not directly comparable for ARIMA (per-product), skip

x = np.arange(len(models))
width = 0.35
bars1 = ax.bar(x - width/2, maes, width, label='MAE (越小越好)', color=C5[0], alpha=0.85)
bars2 = ax.bar(x + width/2, rmses, width, label='RMSE (越小越好)', color=C5[1], alpha=0.85)
for bar, val in zip(bars1, maes):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f'{val:.1f}', ha='center', fontsize=10, fontweight='bold')
for bar, val in zip(bars2, rmses):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f'{val:.1f}', ha='center', fontsize=10, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(models, fontsize=11)
ax.set_ylabel('Error (件)', fontsize=12)
ax.set_title(f'模型对比: 预测误差\nRandom Forest R²={r2_rf:.3f}' +
             (f', XGBoost R²={r2_xgb:.3f}' if HAS_XGB else ''),
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
plt.tight_layout()
fig.savefig(f'{OUT}/prediction_charts/model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('Chart: Model Comparison done')

# ── Chart 5: 30-day Prediction Trend ──
fig, ax = plt.subplots(figsize=(14, 5.5))
df_test_meta['pred_rf'] = y_pred_rf
daily_actual = df_test_meta.groupby('date')['sales'].sum()
daily_pred_rf = df_test_meta.groupby('date')['pred_rf'].sum()
ax.plot(daily_actual.index, daily_actual.values, 'o-', color=C5[0], linewidth=2, markersize=5, label='实际销量')
ax.plot(daily_pred_rf.index, daily_pred_rf.values, 's--', color=C5[1], linewidth=2, markersize=5, label='Random Forest 预测')
if HAS_XGB:
    df_test_meta['pred_xgb'] = y_pred_xgb
    daily_pred_xgb = df_test_meta.groupby('date')['pred_xgb'].sum()
    ax.plot(daily_pred_xgb.index, daily_pred_xgb.values, '^:', color=C5[2], linewidth=2, markersize=5, label='XGBoost 预测')
ax.set_title('12月每日总销量: 实际 vs 预测', fontsize=14, fontweight='bold')
ax.set_xlabel('日期', fontsize=12); ax.set_ylabel('总销量（件）', fontsize=12)
ax.legend(fontsize=10)
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m-%d'))
ax.xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=3))
fig.autofmt_xdate()
plt.tight_layout()
fig.savefig(f'{OUT}/prediction_charts/30day_trend.png', dpi=150, bbox_inches='tight')
plt.close()
print('Chart: 30-Day Trend done')

# ── Chart 6: ARIMA detail (if available) ──
if arima_results:
    fig, axes = plt.subplots(min(3, len(arima_results)), 1, figsize=(14, 4*min(3, len(arima_results))))
    if len(arima_results) == 1:
        axes = [axes]
    for i, res in enumerate(arima_results[:3]):
        ax = axes[i]
        ax.plot(res['dates'], res['actual'], 'o-', color=C5[0], linewidth=2, markersize=5, label='实际')
        ax.plot(res['dates'], res['preds'], 's--', color=C5[1], linewidth=2, markersize=5, label='ARIMA 预测')
        ax.set_title(f"{res['product']} — ARIMA (MAE={res['MAE']:.1f}件)", fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m-%d'))
        fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(f'{OUT}/prediction_charts/arima_detail.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Chart: ARIMA Detail done')

# ═══════════════════════════════════════════
# 7. SAVE METRICS
# ═══════════════════════════════════════════
metrics = {
    'RandomForest': {'MAE': round(mae_rf,1), 'RMSE': round(rmse_rf,1), 'R2': round(r2_rf,4), 'MAPE': round(mape_rf,1)},
}
if HAS_XGB:
    metrics['XGBoost'] = {'MAE': round(mae_xgb,1), 'RMSE': round(rmse_xgb,1), 'R2': round(r2_xgb,4), 'MAPE': round(mape_xgb,1)}
if arima_results:
    metrics['ARIMA_top5_avg'] = {'MAE': round(avg_mae_arima,1), 'RMSE': round(avg_rmse_arima,1), 'R2': None}

metrics['top_features_rf'] = rf_importance.head(10)['feature'].tolist()
metrics['train_rows'] = len(X_train)
metrics['test_rows'] = len(X_test)
metrics['n_features'] = len(FEATURE_COLS)
metrics['split_date'] = str(split_date.date())

with open(f'{OUT}/prediction_metrics.json', 'w') as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 60)
print("7. SUMMARY")
print("=" * 60)
print(f"\n{'Model':<20} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'MAPE':>8}")
print("-" * 55)
print(f"{'Random Forest':<20} {mae_rf:>8.1f} {rmse_rf:>8.1f} {r2_rf:>8.4f} {mape_rf:>7.1f}%")
if HAS_XGB:
    print(f"{'XGBoost':<20} {mae_xgb:>8.1f} {rmse_xgb:>8.1f} {r2_xgb:>8.4f} {mape_xgb:>7.1f}%")
if arima_results:
    print(f"{'ARIMA (avg top5)':<20} {avg_mae_arima:>8.1f} {avg_rmse_arima:>8.1f} {'—':>8} {'—':>7}")

print("\n✅ All done! Charts saved to prediction_charts/")
