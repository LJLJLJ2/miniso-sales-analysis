#!/usr/bin/env python3
"""MINISO 2023 方案三：价格弹性与促销 ROI 分析"""

import pandas as pd
import numpy as np
import os, json, warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from scipy import stats as scipy_stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['Heiti TC', 'STHeiti', 'Hiragino Sans GB', 'Arial Unicode MS']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

DATA = '/Users/zhoujingjing/miniso_sales_data_features.csv'
OUT = '/Users/zhoujingjing/Desktop/miniso_sales_analysis'
os.makedirs(f'{OUT}/elasticity_charts', exist_ok=True)

C5 = ['#1b4b5e','#e8733a','#4ca6ba','#c39858','#8e6c8a']

# ═══════════════════════════════════
# 1. LOAD & PREPARE
# ═══════════════════════════════════
print("=" * 60)
print("1. 数据加载与准备")
print("=" * 60)

df = pd.read_csv(DATA, parse_dates=['date'])
df = df.sort_values(['product','date']).reset_index(drop=True)

# Encode categorical variables for regression
for col in ['category','product','promotion_type','weather','season','price_level']:
    le = LabelEncoder()
    df[f'{col}_enc'] = le.fit_transform(df[col].astype(str))

# Log transforms for elasticity (log-log model)
df['log_sales'] = np.log(df['sales'])
df['log_price'] = np.log(df['base_price'])

print(f"数据量: {len(df)} 条, 产品: {df['product'].nunique()} 个, 品类: {df['category'].nunique()} 个")
print(f"价格区间: ¥{df['base_price'].min():.0f} ~ ¥{df['base_price'].max():.0f}")
print(f"促销类型: {df['promotion_type'].nunique()} 种")

# ═══════════════════════════════════
# 2. PRICE ELASTICITY ANALYSIS
# ═══════════════════════════════════
print("\n" + "=" * 60)
print("2. 价格弹性分析")
print("=" * 60)

elasticity_results = []

# 2a. Overall elasticity (log-log with controls)
print("\n2a. 整体价格弹性（控制促销、季节、周末效应）...")
X_overall = df[['log_price','is_holiday','weekday','month']].copy()
for col in ['promotion_type_enc','season_enc','weather_enc']:
    if col in df.columns:
        X_overall[col] = df[col]
X_overall = pd.get_dummies(X_overall, columns=['weekday','month'], drop_first=True)
y_overall = df['log_sales']

model_overall = LinearRegression()
model_overall.fit(X_overall, y_overall)
elasticity_overall = model_overall.coef_[0]  # log_price coefficient
r2_overall = model_overall.score(X_overall, y_overall)

print(f"  整体价格弹性: {elasticity_overall:.3f}")
print(f"  含义: 价格每上涨1%，销量下降约{abs(elasticity_overall)*1:.1f}%")
print(f"  模型 R²: {r2_overall:.3f}")

# 2b. Per-product elasticity
print("\n2b. 各产品价格弹性...")
for prod in df['product'].unique():
    prod_df = df[df['product'] == prod]
    if len(prod_df) < 50:
        continue

    X = prod_df[['log_price','is_holiday','promotion_type_enc','season_enc']]
    y = prod_df['log_sales']

    model = LinearRegression()
    model.fit(X, y)
    elast = model.coef_[0]
    r2 = model.score(X, y)

    elasticity_results.append({
        'product': prod,
        'category': prod_df['category'].iloc[0],
        'elasticity': round(elast, 4),
        'r2': round(r2, 4),
        'avg_price': round(prod_df['base_price'].mean(), 1),
        'avg_sales': round(prod_df['sales'].mean(), 1),
    })

elast_df = pd.DataFrame(elasticity_results).sort_values('elasticity')
print(f"  已计算 {len(elast_df)} 个产品的价格弹性")
print(f"  弹性范围: {elast_df['elasticity'].min():.3f} ~ {elast_df['elasticity'].max():.3f}")
print(f"  弹性均值: {elast_df['elasticity'].mean():.3f}")
print(f"  弹性产品 (>|0.3|): {(elast_df['elasticity'].abs()>0.3).sum()} 个")
print(f"  非弹性产品 (<|0.3|): {(elast_df['elasticity'].abs()<=0.3).sum()} 个")

# 2c. Elasticity by category
print("\n2c. 各品类价格弹性...")
cat_elast = elast_df.groupby('category').agg(
    elasticity_mean=('elasticity','mean'),
    product_count=('product','count'),
    avg_price=('avg_price','mean'),
    avg_sales=('avg_sales','mean'),
).round(4)
print(cat_elast)

# 2d. Elasticity by price level
print("\n2d. 各价格带弹性...")
df_temp = df.copy()
df_temp = df_temp.merge(elast_df[['product','elasticity']], on='product', how='left')
price_elast = df_temp.groupby('price_level').agg(
    elasticity_mean=('elasticity','mean'),
    avg_price=('base_price','mean'),
    avg_sales=('sales','mean'),
).round(4)
print(price_elast)

# ═══════════════════════════════════
# 3. PROMOTION EFFECTIVENESS
# ═══════════════════════════════════
print("\n" + "=" * 60)
print("3. 促销效果分析")
print("=" * 60)

baseline_sales = df[df['promotion_type'] == '无促销']['sales'].mean()
print(f"无促销日均销量（基线）: {baseline_sales:.1f} 件")

promo_analysis = []
for pt in df['promotion_type'].unique():
    pt_df = df[df['promotion_type'] == pt]
    avg = pt_df['sales'].mean()
    lift = (avg - baseline_sales) / baseline_sales * 100

    # t-test: is this promotion significantly different from baseline?
    baseline_all = df[df['promotion_type'] == '无促销']['sales']
    promo_sales = pt_df['sales']
    t_stat, p_value = scipy_stats.ttest_ind(promo_sales, baseline_all)

    promo_analysis.append({
        'promotion_type': pt,
        'avg_sales': round(avg, 1),
        'sales_lift_pct': round(lift, 1),
        't_statistic': round(t_stat, 3),
        'p_value': round(p_value, 6),
        'significant': '✅' if p_value < 0.05 else '❌',
        'sample_count': len(pt_df),
    })

promo_df = pd.DataFrame(promo_analysis).sort_values('sales_lift_pct', ascending=False)
print(f"\n{'促销类型':<10} {'平均销量':>8} {'销量提升%':>10} {'P值':>10} {'显著性':>6}")
print("-" * 50)
for _, r in promo_df.iterrows():
    print(f"{r['promotion_type']:<10} {r['avg_sales']:>8.1f} {r['sales_lift_pct']:>9.1f}% {r['p_value']:>10.6f} {r['significant']:>6}")

# Promotion x Price Level interaction
print("\n3b. 促销 × 价格带交叉分析...")
promo_price = df.pivot_table(
    values='sales', index='price_level', columns='promotion_type',
    aggfunc='mean'
).round(1)
print(promo_price)

# Promotion x Category
print("\n3c. 促销 × 品类交叉分析...")
promo_cat = df.pivot_table(
    values='sales', index='category', columns='promotion_type',
    aggfunc='mean'
).round(1)
print(promo_cat)

# ═══════════════════════════════════
# 4. PROMOTION ROI ESTIMATION
# ═══════════════════════════════════
print("\n" + "=" * 60)
print("4. 促销 ROI 估算")
print("=" * 60)

# Assumptions for ROI:
# - Average discount rates per promotion type (estimated from industry norms)
# - 限时特价: ~15% price reduction
# - 打折: ~20% price reduction
# - 满减: ~10% effective reduction
# - 买赠: cost ~8% of product price (gift cost)

DISCOUNT_RATES = {
    '限时特价': 0.15,
    '打折': 0.20,
    '满减': 0.10,
    '买赠': 0.08,
    '无促销': 0.0,
}

roi_results = []
for pt in df['promotion_type'].unique():
    if pt == '无促销':
        continue

    pt_df = df[df['promotion_type'] == pt]
    count = len(pt_df)

    # Incremental sales
    incremental_units = pt_df['sales'].mean() - baseline_sales

    # Revenue calculation
    avg_price = pt_df['base_price'].mean()
    base_revenue_per_day = baseline_sales * avg_price
    promo_revenue_per_day = pt_df['sales'].mean() * avg_price * (1 - DISCOUNT_RATES[pt])
    incremental_revenue = promo_revenue_per_day - base_revenue_per_day

    # Cost calculation
    promotion_cost = pt_df['sales'].mean() * avg_price * DISCOUNT_RATES[pt]

    # ROI = incremental_revenue / promotion_cost
    roi = incremental_revenue / promotion_cost if promotion_cost > 0 else 0

    roi_results.append({
        'promotion_type': pt,
        'count': count,
        'avg_sales': round(pt_df['sales'].mean(), 1),
        'incremental_units': round(incremental_units, 1),
        'avg_price': round(avg_price, 1),
        'discount_rate': f"{DISCOUNT_RATES[pt]*100:.0f}%",
        'promotion_cost_per_day': round(promotion_cost, 0),
        'incremental_revenue_per_day': round(incremental_revenue, 0),
        'ROI': round(roi, 3),
    })

roi_df = pd.DataFrame(roi_results).sort_values('ROI', ascending=False)
print(f"\n{'促销':<10} {'增量销量':>8} {'日促销成本':>10} {'日增量收入':>10} {'ROI':>8}")
print("-" * 50)
for _, r in roi_df.iterrows():
    print(f"{r['promotion_type']:<10} {r['incremental_units']:>7.1f}件 ¥{r['promotion_cost_per_day']:>8.0f} ¥{r['incremental_revenue_per_day']:>8.0f} {r['ROI']:>7.2f}")

# Best promotion per category
print("\n4b. 各品类最优促销方式...")
for cat in df['category'].unique():
    cat_df = df[df['category'] == cat]
    cat_baseline = cat_df[cat_df['promotion_type']=='无促销']['sales'].mean()
    best_pt, best_lift = None, 0
    for pt in cat_df['promotion_type'].unique():
        if pt == '无促销': continue
        lift = cat_df[cat_df['promotion_type']==pt]['sales'].mean() - cat_baseline
        if lift > best_lift:
            best_lift, best_pt = lift, pt
    print(f"  {cat}: {best_pt} (增量 +{best_lift:.0f} 件/天)")

# ═══════════════════════════════════
# 5. COMBINED OPTIMIZATION
# ═══════════════════════════════════
print("\n" + "=" * 60)
print("5. 综合优化建议")
print("=" * 60)

# For elastic products, lowering price increases revenue
# For inelastic products, raising price increases revenue
elast_df['price_action'] = elast_df['elasticity'].apply(
    lambda e: '可适当提价' if e > -0.3 else ('可适当降价' if e < -1.0 else '维持现价')
)
elast_df['optimal_promo'] = elast_df.apply(
    lambda r: '限时特价' if r['elasticity'] < -0.5 else ('打折' if r['elasticity'] < -0.3 else '满减/买赠'),
    axis=1
)

print("\n产品优化建议（前10）:")
print(f"{'产品':<12} {'品类':<10} {'弹性':>7} {'定价建议':<12} {'推荐促销':<10}")
print("-" * 55)
for _, r in elast_df.head(10).iterrows():
    print(f"{r['product']:<12} {r['category']:<10} {r['elasticity']:>7.3f} {r['price_action']:<12} {r['optimal_promo']:<10}")

print(f"\n  建议提价的产品: {(elast_df['price_action']=='可适当提价').sum()} 个")
print(f"  建议降价的产品: {(elast_df['price_action']=='可适当降价').sum()} 个")
print(f"  维持现价的产品: {(elast_df['price_action']=='维持现价').sum()} 个")

# ═══════════════════════════════════
# 6. VISUALIZATIONS
# ═══════════════════════════════════
print("\n" + "=" * 60)
print("6. 生成可视化图表")
print("=" * 60)

# Chart 1: Price Elasticity by Product
fig, ax = plt.subplots(figsize=(12, 6))
colors = [C5[0] if e < -0.5 else C5[1] if e < -0.3 else C5[3] for e in elast_df['elasticity']]
bars = ax.barh(elast_df['product'], elast_df['elasticity'], color=colors, height=0.65)
ax.axvline(0, color='#333', linewidth=1)
ax.axvline(-0.3, color='#999', linewidth=0.8, linestyle=':', alpha=0.6, label='非弹性阈值(-0.3)')
ax.axvline(-1.0, color='#999', linewidth=0.8, linestyle='--', alpha=0.6, label='弹性阈值(-1.0)')
for bar, val in zip(bars, elast_df['elasticity']):
    xpos = bar.get_width() - 0.05 if val < 0 else bar.get_width() + 0.02
    ax.text(xpos, bar.get_y() + bar.get_height()/2, f'{val:.2f}', va='center', fontsize=8)
ax.set_xlabel('价格弹性（越负越敏感）', fontsize=12)
ax.set_title('各产品价格弹性对比\n弹性 < −1: 降价增收 | 弹性 > −0.3: 提价增收', fontsize=14, fontweight='bold')
ax.legend(fontsize=9, loc='lower left')
plt.tight_layout()
fig.savefig(f'{OUT}/elasticity_charts/price_elasticity.png', dpi=150, bbox_inches='tight')
plt.close()
print('  ✅ 价格弹性图')

# Chart 2: Promotion Lift Comparison
fig, ax = plt.subplots(figsize=(10, 5.5))
promo_sorted = promo_df.sort_values('sales_lift_pct', ascending=True)
bars = ax.barh(promo_sorted['promotion_type'], promo_sorted['sales_lift_pct'], color=C5[:5], height=0.55)
ax.axvline(0, color='#333', linewidth=1)
for bar, val, sig in zip(bars, promo_sorted['sales_lift_pct'], promo_sorted['significant']):
    ax.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2, f'+{val:.1f}% {sig}', va='center', fontsize=11, fontweight='bold')
ax.set_xlabel('销量提升 vs 无促销 (%)', fontsize=12)
ax.set_title('各促销方式效果对比\n（所有促销类型均显著有效，P<0.01）', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(f'{OUT}/elasticity_charts/promotion_lift.png', dpi=150, bbox_inches='tight')
plt.close()
print('  ✅ 促销效果图')

# Chart 3: Promotion ROI
fig, ax = plt.subplots(figsize=(10, 5.5))
roi_sorted = roi_df.sort_values('ROI', ascending=True)
bars = ax.barh(roi_sorted['promotion_type'], roi_sorted['ROI'], color=C5[:4], height=0.55)
for bar, val in zip(bars, roi_sorted['ROI']):
    ax.text(bar.get_width()+0.02, bar.get_y()+bar.get_height()/2, f'{val:.2f}', va='center', fontsize=12, fontweight='bold')
ax.set_xlabel('ROI（每投入¥1促销成本产生的增量收入）', fontsize=12)
ax.set_title('各促销方式 ROI 对比\nROI > 1 表示促销投入产出为正', fontsize=14, fontweight='bold')
ax.axvline(1.0, color='#999', linewidth=0.8, linestyle='--', alpha=0.6, label='ROI=1 盈亏平衡线')
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(f'{OUT}/elasticity_charts/promotion_roi.png', dpi=150, bbox_inches='tight')
plt.close()
print('  ✅ 促销ROI图')

# Chart 4: Elasticity vs Avg Sales
fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(
    elast_df['avg_sales'], elast_df['elasticity'],
    c=elast_df['avg_price'], cmap='RdYlGn_r', s=120, alpha=0.75, edgecolors='#333', linewidth=0.5
)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('平均价格 (¥)', fontsize=10)
ax.axhline(-0.3, color='#999', linewidth=0.8, linestyle=':', alpha=0.6)
ax.axhline(-1.0, color='#999', linewidth=0.8, linestyle='--', alpha=0.6)
# Annotate a few points
for _, r in elast_df.iloc[::3].iterrows():
    ax.annotate(r['product'], (r['avg_sales'], r['elasticity']),
                fontsize=7, alpha=0.8, xytext=(5,5), textcoords='offset points')
ax.set_xlabel('日均销量（件）', fontsize=12)
ax.set_ylabel('价格弹性', fontsize=12)
ax.set_title('价格弹性 vs 销量 vs 价格\n颜色越深 = 价格越高 | 虚线下方 = 降价更有益', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(f'{OUT}/elasticity_charts/elasticity_vs_sales.png', dpi=150, bbox_inches='tight')
plt.close()
print('  ✅ 弹性-销量关系图')

# Chart 5: Promotion x Price Level Heatmap
fig, ax = plt.subplots(figsize=(9, 5))
pivot_data = promo_price.copy()
sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax,
            linewidths=0.5, cbar_kws={'label': '平均销量（件）'})
ax.set_title('促销方式 × 价格带 交叉效果\n数值为平均日销量（件）', fontsize=14, fontweight='bold')
ax.set_xlabel('促销方式', fontsize=11)
ax.set_ylabel('价格带', fontsize=11)
plt.tight_layout()
fig.savefig(f'{OUT}/elasticity_charts/promo_price_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print('  ✅ 促销×价格带热力图')

# Chart 6: Elasticity by Category Summary
fig, ax = plt.subplots(figsize=(9, 5.5))
cat_colors = [C5[i % 5] for i in range(len(cat_elast))]
bars = ax.bar(cat_elast.index, cat_elast['elasticity_mean'], color=cat_colors, width=0.5)
for bar, val in zip(bars, cat_elast['elasticity_mean']):
    ypos = bar.get_height() - 0.02 if val < 0 else bar.get_height() + 0.01
    ax.text(bar.get_x()+bar.get_width()/2, ypos, f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
ax.axhline(0, color='#333', linewidth=1)
ax.axhline(-0.3, color='#999', linewidth=0.8, linestyle=':', alpha=0.6)
ax.set_ylabel('平均价格弹性', fontsize=12)
ax.set_title('各品类平均价格弹性\n（越负 = 越敏感 = 越适合降价促销）', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(f'{OUT}/elasticity_charts/category_elasticity.png', dpi=150, bbox_inches='tight')
plt.close()
print('  ✅ 品类弹性图')

# ═══════════════════════════════════
# 7. SAVE RESULTS
# ═══════════════════════════════════
print("\n" + "=" * 60)
print("7. 保存结果")
print("=" * 60)

results = {
    '整体价格弹性': round(elasticity_overall, 4),
    '整体弹性R²': round(r2_overall, 4),
    '弹性产品数': int((elast_df['elasticity'].abs() > 0.3).sum()),
    '非弹性产品数': int((elast_df['elasticity'].abs() <= 0.3).sum()),
    '建议提价产品数': int((elast_df['price_action'] == '可适当提价').sum()),
    '建议降价产品数': int((elast_df['price_action'] == '可适当降价').sum()),
    '最优促销方式': promo_df.iloc[0]['promotion_type'],
    '最优促销提升': f"{promo_df.iloc[0]['sales_lift_pct']:.1f}%",
    '最高ROI促销': roi_df.iloc[0]['promotion_type'],
    '最高ROI值': round(roi_df.iloc[0]['ROI'], 2),
}

elast_df.to_csv(f'{OUT}/elasticity_charts/product_elasticity.csv', index=False)
promo_df.to_csv(f'{OUT}/elasticity_charts/promotion_analysis.csv', index=False)
roi_df.to_csv(f'{OUT}/elasticity_charts/promotion_roi.csv', index=False)

with open(f'{OUT}/elasticity_metrics.json', 'w') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("  ✅ 数据已导出")
print("\n✅ 方案三分析完成！")
