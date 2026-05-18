#!/usr/bin/env python3
"""MINISO 2023 Sales — Full Analysis (pandas + SQL + SciPy + Seaborn + matplotlib)"""

import pandas as pd
import numpy as np
import sqlite3
import os, warnings
warnings.filterwarnings('ignore')

# ── Visualization ──
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# ── Statistics ──
from scipy import stats
from scipy.stats import f_oneway, pearsonr, ttest_ind

# ── Setup ──
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['Heiti TC', 'STHeiti', 'Hiragino Sans GB', 'Arial Unicode MS']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

DATA = '/Users/zhoujingjing/miniso_sales_data.csv'
OUT  = '/Users/zhoujingjing/Desktop/miniso_sales_analysis'
os.makedirs(f'{OUT}/charts', exist_ok=True)

C5 = ['#1b4b5e','#e8733a','#4ca6ba','#c39858','#8e6c8a']
C3 = ['#1b4b5e','#e8733a','#4ca6ba']

# ═══════════════════════════════════════════════════════
# 1. LOAD & PREP
# ═══════════════════════════════════════════════════════
df = pd.read_csv(DATA, parse_dates=['date'])
df['revenue']    = df['sales_amount']
df['asp']        = df['revenue'] / df['sales']
df['month_name'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek  # 0=Mon

print(f"Data: {len(df)} rows, {df['date'].min().date()} ~ {df['date'].max().date()}")
print(f"Total revenue: ¥{df['revenue'].sum():,.0f} | Total units: {df['sales'].sum():,}")

# ═══════════════════════════════════════════════════════
# 2. SQL ANALYSIS (SQLite)
# ═══════════════════════════════════════════════════════
print("\n" + "="*60)
print("SQL ANALYSIS (SQLite)")
print("="*60)

conn = sqlite3.connect(':memory:')
df.to_sql('sales', conn, index=False, if_exists='replace')

sql_queries = {
    "月度营收与销量": """
        SELECT month,
               ROUND(SUM(revenue)/10000, 1) AS revenue_wan,
               SUM(sales) AS total_units,
               ROUND(AVG(asp), 1) AS avg_price
        FROM sales GROUP BY month ORDER BY month
    """,
    "品类表现": """
        SELECT category,
               ROUND(SUM(revenue)/10000, 1) AS revenue_wan,
               SUM(sales) AS total_units,
               ROUND(SUM(revenue)*100.0/(SELECT SUM(revenue) FROM sales), 1) AS pct,
               ROUND(AVG(turnover_rate)*100, 1) AS avg_turnover_pct
        FROM sales GROUP BY category ORDER BY revenue_wan DESC
    """,
    "促销效果对比": """
        SELECT promotion_type,
               COUNT(*) AS records,
               ROUND(AVG(sales), 1) AS avg_daily_units,
               ROUND(SUM(revenue)/10000, 1) AS total_revenue_wan
        FROM sales GROUP BY promotion_type ORDER BY avg_daily_units DESC
    """,
    "价格档位 × 促销交叉分析": """
        SELECT price_level, promotion_type,
               ROUND(AVG(sales), 1) AS avg_units,
               COUNT(*) AS n
        FROM sales GROUP BY price_level, promotion_type
        ORDER BY price_level, avg_units DESC
    """,
    "天气对销量的影响": """
        SELECT weather,
               ROUND(AVG(sales), 1) AS avg_units,
               ROUND(AVG(revenue), 1) AS avg_revenue,
               COUNT(*) AS records
        FROM sales GROUP BY weather ORDER BY avg_units DESC
    """,
    "Top 5 产品 × 最佳月份": """
        SELECT product, category, month,
               SUM(sales) AS total_units,
               ROUND(SUM(revenue)/10000, 1) AS revenue_wan
        FROM sales
        WHERE product IN (SELECT product FROM sales GROUP BY product ORDER BY SUM(revenue) DESC LIMIT 5)
        GROUP BY product, month ORDER BY product, month
    """,
}

for name, sql in sql_queries.items():
    result = pd.read_sql(sql, conn)
    print(f"\n── {name} ──")
    print(result.head(12).to_string(index=False))

conn.close()
print("\nSQL analysis complete — 6 queries executed")

# ═══════════════════════════════════════════════════════
# 3. STATISTICAL TESTS (SciPy)
# ═══════════════════════════════════════════════════════
print("\n" + "="*60)
print("STATISTICAL TESTS (SciPy)")
print("="*60)

# 3a. T-test: 促销 vs 无促销
promo_mask = df['promotion_type'] != '无促销'
promo_sales = df[promo_mask]['sales']
no_promo_sales = df[~promo_mask]['sales']
t_stat, p_val = ttest_ind(promo_sales, no_promo_sales, equal_var=False)
print(f"\n1. T-test: 有促销 vs 无促销 日均销量")
print(f"   有促销均值: {promo_sales.mean():.1f} | 无促销均值: {no_promo_sales.mean():.1f}")
print(f"   t = {t_stat:.2f}, p = {p_val:.6f} {'*** 极显著' if p_val < 0.001 else '* 显著' if p_val < 0.05 else '  不显著'}")

# 3b. ANOVA: 品类间销售差异
cat_groups = [g['sales'].values for _, g in df.groupby('category')]
f_cat, p_cat = f_oneway(*cat_groups)
print(f"\n2. ANOVA: 品类间销售差异")
for cat, grp in df.groupby('category')['sales']:
    print(f"   {cat}: mean={grp.mean():.1f}, std={grp.std():.1f}")
print(f"   F = {f_cat:.2f}, p = {p_cat:.6f} {'*** 极显著' if p_cat < 0.001 else '* 显著' if p_cat < 0.05 else '  不显著'}")

# 3c. ANOVA: 促销类型间差异 (仅促销)
promo_groups = [g['sales'].values for _, g in df[df['promotion_type']!='无促销'].groupby('promotion_type')]
f_promo, p_promo = f_oneway(*promo_groups)
print(f"\n3. ANOVA: 各促销类型间差异")
for pt, grp in df[df['promotion_type']!='无促销'].groupby('promotion_type')['sales']:
    print(f"   {pt}: mean={grp.mean():.1f}, std={grp.std():.1f}")
print(f"   F = {f_promo:.2f}, p = {p_promo:.6f} {'*** 极显著' if p_promo < 0.001 else '* 显著' if p_promo < 0.05 else '  不显著'}")

# 3d. ANOVA: 天气影响
weather_groups = [g['sales'].values for _, g in df.groupby('weather')]
f_w, p_w = f_oneway(*weather_groups)
print(f"\n4. ANOVA: 天气对销量影响")
for w, grp in df.groupby('weather')['sales']:
    print(f"   {w}: mean={grp.mean():.1f}")
print(f"   F = {f_w:.2f}, p = {p_w:.6f} {'*** 极显著' if p_w < 0.001 else '* 显著' if p_w < 0.05 else '  不显著'}")

# 3e. Correlation: 价格 vs 销量
r_price, p_price = pearsonr(df['base_price'], df['sales'])
print(f"\n5. Pearson 相关系数: 价格 vs 销量")
print(f"   r = {r_price:.4f}, p = {p_price:.6f} {'*** 极显著' if p_price < 0.001 else '* 显著' if p_price < 0.05 else '  不显著'}")

# 3f. T-test: Holiday effect
holiday_sales = df[df['is_holiday']==True]['sales']
normal_sales   = df[df['is_holiday']==False]['sales']
t_h, p_h = ttest_ind(holiday_sales, normal_sales, equal_var=False)
print(f"\n6. T-test: 节假日 vs 非节假日")
print(f"   节假日均值: {holiday_sales.mean():.1f} | 非节假日均值: {normal_sales.mean():.1f}")
print(f"   t = {t_h:.2f}, p = {p_h:.6f} {'*** 极显著' if p_h < 0.001 else '* 显著' if p_h < 0.05 else '  不显著'}")

# ═══════════════════════════════════════════════════════
# 4. ENHANCED VISUALIZATIONS (Seaborn + matplotlib)
# ═══════════════════════════════════════════════════════
print("\n" + "="*60)
print("VISUALIZATIONS")
print("="*60)

# ── Chart A: Seaborn heatmap — 品类×月份 营收 ──
pivot = df.pivot_table(values='revenue', index='category', columns='month', aggfunc='sum') / 1e4
fig, ax = plt.subplots(figsize=(14, 5.5))
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', linewidths=0.8,
            cbar_kws={'label': 'Revenue (¥10k)'}, ax=ax, annot_kws={'fontsize':8})
ax.set_title('Revenue Heatmap: Category × Month (¥10k)', fontsize=14, fontweight='bold')
ax.set_xlabel('Month', fontsize=12); ax.set_ylabel('Category', fontsize=12)
plt.tight_layout()
fig.savefig(f'{OUT}/charts/09_heatmap_category_month.png', dpi=150, bbox_inches='tight')
plt.close()
print('Chart 9/13: Heatmap done')

# ── Chart B: Seaborn boxplot — 促销类型销量分布 ──
fig, ax = plt.subplots(figsize=(12, 5))
order = df.groupby('promotion_type')['sales'].mean().sort_values(ascending=False).index.tolist()
palette = [C5[0] if p != '无促销' else '#aaa' for p in order]
sns.boxplot(data=df, x='promotion_type', y='sales', order=order, palette=palette,
            width=0.55, linewidth=0.8, fliersize=2, ax=ax)
ax.set_title('Sales Distribution by Promotion Type', fontsize=14, fontweight='bold')
ax.set_xlabel('Promotion Type', fontsize=12); ax.set_ylabel('Daily Units per SKU', fontsize=12)
# Add mean labels
means = df.groupby('promotion_type')['sales'].mean()
for i, pt in enumerate(order):
    ax.text(i, means[pt]+3, f'μ={means[pt]:.0f}', ha='center', fontsize=9, fontweight='bold', color=C5[0])
plt.tight_layout()
fig.savefig(f'{OUT}/charts/10_boxplot_promotion.png', dpi=150, bbox_inches='tight')
plt.close()
print('Chart 10/13: Boxplot done')

# ── Chart C: Seaborn barplot — 品类×价格档位 ──
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=df, x='category', y='sales', hue='price_level', estimator=np.mean,
            palette=C3, ax=ax, ci=None)
ax.set_title('Avg Daily Sales: Category × Price Level', fontsize=14, fontweight='bold')
ax.set_xlabel('Category', fontsize=12); ax.set_ylabel('Avg Daily Units', fontsize=12)
ax.legend(title='Price Level', fontsize=9)
plt.tight_layout()
fig.savefig(f'{OUT}/charts/11_barplot_category_price.png', dpi=150, bbox_inches='tight')
plt.close()
print('Chart 11/13: Barplot done')

# ── Chart D: Correlation heatmap ──
corr_cols = ['sales','revenue','base_price','inventory','turnover_rate','month','weekday']
corr = df[corr_cols].corr()
fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            linewidths=0.5, square=True, ax=ax, vmin=-1, vmax=1)
ax.set_title('Correlation Matrix of Key Variables', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(f'{OUT}/charts/12_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print('Chart 12/13: Correlation done')

# ── Chart E: Seaborn violin plot — 季节×销量 ──
season_order = ['春季','夏季','秋季','冬季']
fig, ax = plt.subplots(figsize=(11, 5))
sns.violinplot(data=df, x='season', y='sales', order=season_order,
               palette=['#7ec87b','#e8733a','#c39858','#6b9ec4'], ax=ax, inner='quartile')
ax.set_title('Sales Distribution by Season (Violin Plot)', fontsize=14, fontweight='bold')
ax.set_xlabel('Season', fontsize=12); ax.set_ylabel('Daily Units per SKU', fontsize=12)
plt.tight_layout()
fig.savefig(f'{OUT}/charts/13_violin_season.png', dpi=150, bbox_inches='tight')
plt.close()
print('Chart 13/13: Violin done')

# ── Original 8 Charts ──
print("Generating original 8 charts...")

# Chart 1: Daily trend with 7-day MA
fig, ax1 = plt.subplots(figsize=(16, 5.5))
daily = df.groupby('date').agg(revenue=('revenue','sum'), sales=('sales','sum')).reset_index()
daily['revenue_ma7'] = daily['revenue'].rolling(7).mean()
daily['sales_ma7'] = daily['sales'].rolling(7).mean()
ax1.fill_between(daily['date'], daily['revenue'], alpha=0.25, color=C5[0])
ax1.plot(daily['date'], daily['revenue_ma7'], color=C5[0], linewidth=1.8, label='Revenue (7-day MA)')
ax1.set_ylabel('Revenue (¥)', fontsize=12); ax1.set_xlabel('Date', fontsize=12)
ax2 = ax1.twinx()
ax2.plot(daily['date'], daily['sales_ma7'], color=C5[1], linewidth=1.8, label='Sales units (7-day MA)')
ax2.set_ylabel('Sales Units', fontsize=12)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, loc='upper left', fontsize=10)
ax1.set_title('Daily Revenue & Sales Trend (7-day Moving Average)', fontsize=14, fontweight='bold')
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'¥{x/1e4:.0f}万'))
plt.tight_layout()
fig.savefig(f'{OUT}/charts/01_daily_trend.png', dpi=150, bbox_inches='tight')
plt.close()
print('Chart 1/13: Daily trend done')

# Chart 2: Monthly revenue & sales
monthly = df.groupby('month').agg(revenue=('revenue','sum'), sales=('sales','sum')).reset_index()
monthly['revenue_wan'] = monthly['revenue'] / 1e4
fig, ax1 = plt.subplots(figsize=(12, 5))
bars = ax1.bar(monthly['month'], monthly['revenue_wan'], color=C5[0], alpha=0.85, label='Revenue (¥10k)')
ax1.set_ylabel('Revenue (¥10k)', fontsize=12); ax1.set_xlabel('Month', fontsize=12)
for bar, val in zip(bars, monthly['revenue_wan']):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+8, f'{val:.0f}', ha='center', fontsize=9, fontweight='bold')
ax2 = ax1.twinx()
ax2.plot(monthly['month'], monthly['sales'], 'o-', color=C5[1], linewidth=2, markersize=8, label='Sales Units')
ax2.set_ylabel('Sales Units', fontsize=12)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, loc='upper left', fontsize=10)
ax1.set_title('Monthly Revenue & Sales', fontsize=14, fontweight='bold')
ax1.set_xticks(range(1,13))
plt.tight_layout()
fig.savefig(f'{OUT}/charts/02_monthly.png', dpi=150, bbox_inches='tight')
plt.close()
print('Chart 2/13: Monthly done')

# Chart 3: Category performance
cat = df.groupby('category').agg(revenue=('revenue','sum'), sales=('sales','sum'), asp=('asp','mean')).reset_index()
cat['revenue_wan'] = cat['revenue'] / 1e4
cat = cat.sort_values('revenue_wan', ascending=True)
fig, ax = plt.subplots(figsize=(10, 5.5))
bars = ax.barh(cat['category'], cat['revenue_wan'], color=[C5[i%5] for i in range(len(cat))], height=0.6)
for bar, val in zip(bars, cat['revenue_wan']):
    ax.text(bar.get_width()+15, bar.get_y()+bar.get_height()/2, f'¥{val:.0f}万', va='center', fontsize=10, fontweight='bold')
ax.set_xlabel('Revenue (¥10k)', fontsize=12)
ax.set_title('Revenue by Category', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(f'{OUT}/charts/03_category.png', dpi=150, bbox_inches='tight')
plt.close()
print('Chart 3/13: Category done')

# Chart 4: Top 10 products
prod = df.groupby('product').agg(revenue=('revenue','sum'), sales=('sales','sum')).reset_index()
prod['revenue_wan'] = prod['revenue'] / 1e4
top10 = prod.nlargest(10, 'revenue_wan').sort_values('revenue_wan', ascending=True)
fig, ax = plt.subplots(figsize=(10, 5.5))
colors = [C5[0] if i < 7 else C5[1] for i in range(10)]
bars = ax.barh(top10['product'], top10['revenue_wan'], color=colors, height=0.6)
for bar, val in zip(bars, top10['revenue_wan']):
    ax.text(bar.get_width()+8, bar.get_y()+bar.get_height()/2, f'¥{val:.0f}万', va='center', fontsize=9, fontweight='bold')
ax.set_xlabel('Revenue (¥10k)', fontsize=12)
ax.set_title('Top 10 Products by Revenue', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(f'{OUT}/charts/04_products.png', dpi=150, bbox_inches='tight')
plt.close()
print('Chart 4/13: Products done')

# Chart 5: Promotion type comparison
promo = df.groupby('promotion_type').agg(revenue=('revenue','sum'), sales=('sales','mean'), count=('sales','count')).reset_index()
promo['revenue_wan'] = promo['revenue'] / 1e4
promo = promo.sort_values('sales', ascending=True)
fig, ax = plt.subplots(figsize=(10, 5))
colors = [C5[0] if p != '无促销' else '#aaa' for p in promo['promotion_type']]
bars = ax.barh(promo['promotion_type'], promo['sales'], color=colors, height=0.6)
for bar, val, rev in zip(bars, promo['sales'], promo['revenue_wan']):
    ax.text(bar.get_width()+1.5, bar.get_y()+bar.get_height()/2, f'{val:.0f} units/day | ¥{rev:.0f}万', va='center', fontsize=9)
ax.set_xlabel('Avg Daily Units per SKU', fontsize=12)
ax.set_title('Promotion Type: Avg Daily Sales & Total Revenue', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(f'{OUT}/charts/05_promotion.png', dpi=150, bbox_inches='tight')
plt.close()
print('Chart 5/13: Promotion done')

# Chart 6: Seasonality (season + weekday)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
season_order = ['春季','夏季','秋季','冬季']
colors_s = ['#7ec87b','#e8733a','#c39858','#6b9ec4']
season_data = df.groupby('season').agg(revenue=('revenue','sum')).reindex(season_order).reset_index()
ax1.bar(season_data['season'].astype(str), season_data['revenue']/1e4, color=colors_s)
for i, row in season_data.iterrows():
    ax1.text(i, row['revenue']/1e4+20, f'¥{row["revenue"]/1e4:.0f}万', ha='center', fontsize=10, fontweight='bold')
ax1.set_title('Revenue by Season', fontsize=13, fontweight='bold')
ax1.set_ylabel('Revenue (¥10k)', fontsize=11)
dow_data = df.groupby('day_of_week').agg(sales=('sales','mean')).reset_index()
dow_labels = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
dow_data['day'] = dow_data['day_of_week'].map(lambda x: dow_labels[x])
ax2.bar(dow_data['day'], dow_data['sales'], color=C5[2])
for _, row in dow_data.iterrows():
    ax2.text(row['day'], row['sales']+1, f'{row["sales"]:.0f}', ha='center', fontsize=10, fontweight='bold')
ax2.set_title('Avg Daily Sales by Day of Week', fontsize=13, fontweight='bold')
ax2.set_ylabel('Avg Units per SKU', fontsize=11)
plt.tight_layout()
fig.savefig(f'{OUT}/charts/06_seasonality.png', dpi=150, bbox_inches='tight')
plt.close()
print('Chart 6/13: Seasonality done')

# Chart 7: Weather & Holiday impact
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
weather_order = ['晴天','阴天','雨天','雪天']
weather_colors = ['#f4a742','#8fa4b8','#5b8cb8','#c8d6e5']
weather_data = df.groupby('weather').agg(sales=('sales','mean')).reindex(weather_order).reset_index()
ax1.bar(weather_data['weather'].astype(str), weather_data['sales'], color=weather_colors)
for i, row in weather_data.iterrows():
    ax1.text(i, row['sales']+1, f'{row["sales"]:.0f}', ha='center', fontsize=10, fontweight='bold')
ax1.set_title('Avg Sales by Weather', fontsize=13, fontweight='bold')
ax1.set_ylabel('Avg Units per SKU', fontsize=11)
holiday_data = df.groupby('is_holiday').agg(sales=('sales','mean')).reset_index()
holiday_data['label'] = holiday_data['is_holiday'].map({True:'Holiday', False:'Non-Holiday'})
ax2.bar(holiday_data['label'], holiday_data['sales'], color=[C5[1], C5[0]])
for _, row in holiday_data.iterrows():
    ax2.text(row['label'], row['sales']+1, f'{row["sales"]:.0f}', ha='center', fontsize=10, fontweight='bold')
ax2.set_title('Avg Sales: Holiday vs Non-Holiday', fontsize=13, fontweight='bold')
ax2.set_ylabel('Avg Units per SKU', fontsize=11)
plt.tight_layout()
fig.savefig(f'{OUT}/charts/07_weather_holiday.png', dpi=150, bbox_inches='tight')
plt.close()
print('Chart 7/13: Weather & Holiday done')

# Chart 8: Price level & turnover rate
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
price_order = ['低价','中价','高价']
price_colors = [C5[2], C5[0], C5[1]]
price_data = df.groupby('price_level').agg(revenue=('revenue','sum')).reindex(price_order).reset_index()
wedges, texts, autotexts = ax1.pie(price_data['revenue']/1e4, labels=price_data['price_level'].astype(str),
    colors=price_colors, autopct='%1.1f%%', explode=(0,0,0.05), startangle=90, textprops={'fontsize':11})
for t in autotexts: t.set_fontweight('bold')
ax1.set_title('Revenue Share by Price Level', fontsize=13, fontweight='bold')
turnover = df.groupby('category').agg(turnover=('turnover_rate','mean')).reset_index()
turnover['turnover_pct'] = turnover['turnover'] * 100
turnover = turnover.sort_values('turnover_pct', ascending=True)
ax2.barh(turnover['category'], turnover['turnover_pct'], color=C5[:5], height=0.6)
for _, row in turnover.iterrows():
    ax2.text(row['turnover_pct']+0.5, row['category'], f'{row["turnover_pct"]:.1f}%', va='center', fontsize=10, fontweight='bold')
ax2.set_title('Avg Turnover Rate by Category', fontsize=13, fontweight='bold')
ax2.set_xlabel('Turnover Rate (%)', fontsize=11)
plt.tight_layout()
fig.savefig(f'{OUT}/charts/08_price_turnover.png', dpi=150, bbox_inches='tight')
plt.close()
print('Chart 8/13: Price & Turnover done')


def write_analysis_report(df, out_dir, data_path):
    """从当前 DataFrame 自动生成 分析报告.md，避免手写数字与数据不一致。"""
    from datetime import datetime

    rev = df['revenue']
    tr = float(rev.sum())
    ts = int(df['sales'].sum())
    n = len(df)
    dmin, dmax = df['date'].min().date(), df['date'].max().date()
    n_cat = df['category'].nunique()
    n_prod = df['product'].nunique()
    days_in_year = max((dmax - dmin).days + 1, 1)
    daily_avg_rev = tr / days_in_year

    # ── 月度营收（万元）──
    monthly = df.groupby('month', as_index=False)['revenue'].sum()
    monthly['wan'] = monthly['revenue'] / 1e4
    m_wan = dict(zip(monthly['month'].astype(int), monthly['wan']))
    peak_m = int(monthly.loc[monthly['revenue'].idxmax(), 'month'])
    trough_m = int(monthly.loc[monthly['revenue'].idxmin(), 'month'])
    peak_wan = float(monthly['wan'].max())
    trough_wan = float(monthly['wan'].min())
    peak_trough_pct = (peak_wan - trough_wan) / trough_wan * 100 if trough_wan else 0.0

    h2_rev = float(df.loc[df['month'] >= 7, 'revenue'].sum())
    h2_pct = h2_rev / tr * 100 if tr else 0.0

    # ── 品类 ──
    cat = df.groupby('category', as_index=False).agg(
        revenue=('revenue', 'sum'), sales=('sales', 'sum'))
    cat['wan'] = cat['revenue'] / 1e4
    cat['pct'] = cat['revenue'] / tr * 100
    cat['asp'] = cat['revenue'] / cat['sales']
    cat = cat.sort_values('revenue', ascending=False)

    # ── 产品 Top / Bottom ──
    prod = df.groupby('product', as_index=False)['revenue'].sum()
    prod['wan'] = prod['revenue'] / 1e4
    top3 = prod.nlargest(3, 'revenue')
    bot3 = prod.nsmallest(3, 'revenue')
    top10_share = prod.nlargest(10, 'revenue')['revenue'].sum() / tr * 100 if tr else 0.0

    # ── 促销 ──
    base = df.loc[df['promotion_type'] == '无促销', 'sales'].mean()
    promo = df.groupby('promotion_type', as_index=False).agg(
        avg_sales=('sales', 'mean'), revenue=('revenue', 'sum'))
    promo['rev_pct'] = promo['revenue'] / tr * 100
    promo['lift_pct'] = (promo['avg_sales'] / base - 1) * 100
    promo = promo.sort_values('avg_sales', ascending=False)

    # ── 季节 ──
    season_order = ['春季', '夏季', '秋季', '冬季']
    se = df.groupby('season', as_index=False)['revenue'].sum()
    se['wan'] = se['revenue'] / 1e4
    se['pct'] = se['revenue'] / tr * 100
    se['_ord'] = se['season'].map({s: i for i, s in enumerate(season_order)})
    se = se.sort_values('_ord', na_position='last')

    # ── 工作日 / 周末（与作图一致：0=周一）──
    wk_mask = df['day_of_week'] < 5
    mean_wk = df.loc[wk_mask, 'sales'].mean()
    mean_we = df.loc[~wk_mask, 'sales'].mean()
    weekend_lower_pct = (1 - mean_we / mean_wk) * 100 if mean_wk else 0.0

    # ── 天气 ──
    wx = df.groupby('weather', as_index=False)['sales'].mean()
    wx = wx.sort_values('sales', ascending=False)

    # ── 节假日 ──
    hol_mask = df['is_holiday'].astype(str).str.lower().isin(('true', '1', 'yes'))
    hol = df.loc[hol_mask, 'sales'].mean()
    nhol = df.loc[~hol_mask, 'sales'].mean()
    hol_lift_pct = (hol / nhol - 1) * 100 if nhol else 0.0

    # ── 价格档位 ──
    pl_order = ['低价', '中价', '高价']
    pl = df.groupby('price_level', as_index=False)['revenue'].sum()
    pl['wan'] = pl['revenue'] / 1e4
    pl['pct'] = pl['revenue'] / tr * 100
    pl['_ord'] = pl['price_level'].map({p: i for i, p in enumerate(pl_order)})
    pl = pl.sort_values('_ord')

    # ── 周转 ──
    turn = df.groupby('category', as_index=False)['turnover_rate'].mean()
    turn['turnover_pct'] = turn['turnover_rate'] * 100
    turn = turn.sort_values('turnover_pct', ascending=False)

    # ── 自检 ──
    sum_month = monthly['revenue'].sum()
    sum_cat = cat['revenue'].sum()
    sum_season = se['revenue'].sum()
    checks = [
        ('月度营收之和 = 总营收', abs(sum_month - tr) < 0.01),
        ('品类营收之和 = 总营收', abs(sum_cat - tr) < 0.01),
        ('季节营收之和 = 总营收', abs(sum_season - tr) < 0.01),
        ('品类营收占比之和 ≈ 100%', abs(cat['pct'].sum() - 100) < 0.05),
        ('促销类型营收占比之和 ≈ 100%', abs(promo['rev_pct'].sum() - 100) < 0.05),
    ]
    check_lines = '\n'.join(
        f"- [{'x' if ok else ' '}] {name}" for name, ok in checks)

    def md_table(headers, rows):
        h = '| ' + ' | '.join(headers) + ' |'
        sep = '|' + '|'.join(['---'] * len(headers)) + '|'
        body = '\n'.join('| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows)
        return '\n'.join([h, sep, body])

    month_names = {1: '1月', 2: '2月', 3: '3月', 4: '4月', 5: '5月', 6: '6月',
                   7: '7月', 8: '8月', 9: '9月', 10: '10月', 11: '11月', 12: '12月'}
    monthly_rows = []
    for _, r in monthly.sort_values('month').iterrows():
        mo = int(r['month'])
        monthly_rows.append([month_names[mo], f"{r['wan']:.0f}"])

    cat_rows = [[row['category'], f"{row['wan']:.0f}", f"{row['pct']:.1f}%", f"{row['asp']:.1f}"]
                for _, row in cat.iterrows()]

    promo_rows = []
    for _, row in promo.iterrows():
        lift = '—' if row['promotion_type'] == '无促销' else f"+{row['lift_pct']:.1f}%"
        promo_rows.append([
            row['promotion_type'],
            f"{row['avg_sales']:.1f}",
            lift,
            f"{row['rev_pct']:.1f}%",
        ])

    se_rows = [[row['season'], f"{row['wan']:.0f}", f"{row['pct']:.1f}%"] for _, row in se.iterrows()]

    wx_rows = [[row['weather'], f"{row['sales']:.1f}"] for _, row in wx.iterrows()]

    pl_rows = [[row['price_level'], f"{row['wan']:.0f}", f"{row['pct']:.1f}%"] for _, row in pl.iterrows()]

    turn_rows = [[row['category'], f"{row['turnover_pct']:.1f}%"] for _, row in turn.iterrows()]

    top_str = '、'.join([f"{r['product']}（¥{r['wan']:.0f}万）" for _, r in top3.iterrows()])
    bot_str = '、'.join([f"{r['product']}（¥{r['wan']:.0f}万）" for _, r in bot3.iterrows()])

    # ── 用于「解读」段落的派生指标（仍全部来自 df）──
    top_cat = cat.iloc[0]['category']
    top_cat_pct = float(cat.iloc[0]['pct'])
    top3_cat_pct = float(cat.head(3)['pct'].sum())
    asp_max_cat = cat.loc[cat['asp'].idxmax(), 'category']
    asp_min_cat = cat.loc[cat['asp'].idxmin(), 'category']
    asp_max_v = float(cat['asp'].max())
    asp_min_v = float(cat['asp'].min())
    wx_hi_name = str(wx.iloc[0]['weather'])
    wx_hi_v = float(wx.iloc[0]['sales'])
    wx_lo_name = str(wx.iloc[-1]['weather'])
    wx_lo_v = float(wx.iloc[-1]['sales'])
    wx_rel_spread = (wx_hi_v - wx_lo_v) / wx_lo_v * 100 if wx_lo_v else 0.0
    best_promo_name = str(promo.iloc[0]['promotion_type'])
    best_promo_lift = float(promo.iloc[0]['lift_pct'])
    no_promo_rev_pct = float(promo.loc[promo['promotion_type'] == '无促销', 'rev_pct'].values[0])
    turn_fast = turn.iloc[0]['category']
    turn_fast_v = float(turn.iloc[0]['turnover_pct'])
    turn_slow = turn.iloc[-1]['category']
    turn_slow_v = float(turn.iloc[-1]['turnover_pct'])
    hi_price = pl.loc[pl['price_level'] == '高价'].iloc[0]
    lo_price = pl.loc[pl['price_level'] == '低价'].iloc[0]
    hol_day_share = float(hol_mask.mean() * 100)
    peak_month_cn = month_names[peak_m]
    trough_month_cn = month_names[trough_m]
    if trough_m == 2:
        trough_interpret = (
            f"{trough_month_cn}为全年营收低谷，**常见解释**包括春节假期带来的有效营业日减少、"
            "客流结构变化等（本数据集未含「营业日数」字段，**需与门店实际排班/放假表交叉验证**）。"
        )
    else:
        trough_interpret = (
            f"{trough_month_cn}为相对低谷，建议对照当月**节假日密度、天气、促销排期**做逐周复盘，"
            "避免仅凭单月总值下结论。"
        )
    if peak_m in (7, 8, 11, 12):
        peak_interpret = (
            f"{peak_month_cn}为峰值月份之一，**可能与**暑期/年末消费旺季、平台大促或企业集中备货有关；"
            "若 `promotion_type` 在该月显著升高，则「促销驱动」成分更强（可用 `analysis.py` 里 SQL 按月交叉验证）。"
        )
    else:
        peak_interpret = (
            f"{peak_month_cn}为营收最高月，建议结合**当月促销强度、节假日与品类结构**拆解贡献来源。"
        )

    md = f"""# MINISO 2023 销售数据分析与促销效果评估

> 本报告由 `analysis.py` **根据同一次加载的 DataFrame 自动生成**，与数据文件 `{data_path}` 及图表一致。含**数据解读**小节：其中「可能成因」为结合零售常识的**假设性解释**，非严格因果结论。生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}

---

## 一、项目概述

对 MINISO（名创优品）2023 年销售明细进行汇总分析，涵盖销售趋势、品类结构、产品表现、促销效果、季节性及外部因素等。

### 核心数据

| 指标 | 数值 |
|------|------|
| 数据时间范围 | {dmin} — {dmax} |
| 数据行数 | {n:,} 条 |
| 品类数 | {n_cat} 个 |
| 产品数 | {n_prod} 个 |
| 全年总营收 | ¥{tr:,.0f} |
| 全年总销量 | {ts:,} 件 |
| 日均营收（按自然日） | ¥{daily_avg_rev:,.0f} |

---

## 二、销售趋势分析

![Daily Trend](charts/01_daily_trend.png)

### 2.1 整体走势

全年按日聚合的营收与销量见上图（含 7 日滑动平均）。

### 2.2 月度分布

![Monthly](charts/02_monthly.png)

{md_table(['月份', '营收（万元）'], monthly_rows)}

**要点（由数据自动计算）**：{peak_m}月营收最高（约 **¥{peak_wan:.0f}万**），{trough_m}月最低（约 **¥{trough_wan:.0f}万**），峰谷相对差约 **{peak_trough_pct:.1f}%**。7–12 月营收占全年约 **{h2_pct:.1f}%**。

### 2.3 数据解读与可能成因（非因果证明）

- **峰谷月份**：{peak_interpret}
- **低谷月份**：{trough_interpret}
- **下半年占比 {h2_pct:.1f}%**：若显著高于 50%，**可能反映** Q3/Q4 促销更密、品类结构季节性更强，或样本期内下半年门店/渠道权重变化；**需排除**「仅记录天数不均」等数据构造问题（本表为逐日 SKU 级记录，仍建议核对是否含闭店日）。
- **与趋势图的关系**：日度折线若呈台阶式跳变，往往对应**大促起止**或**节假日**；若呈缓慢爬升，更像**品类/客单结构**迁移。

---

## 三、品类与产品分析

![Category](charts/03_category.png)

### 3.1 品类结构

{md_table(['品类', '营收（万元）', '营收占比', '平均售价（¥）'], cat_rows)}

**要点**：前三大品类营收占比合计 **{top3_cat_pct:.1f}%**。

### 3.1.1 数据解读与可能成因

- **结构集中度**：第一名 **{top_cat}** 约占 **{top_cat_pct:.1f}%** 营收，说明当前盘子的「支柱品类」明确；若头部品类 ASP 同时偏高，则总营收更依赖**高客单而非纯走量**。
- **客单价差异**：品类间 ASP 最高为 **{asp_max_cat}（约 ¥{asp_max_v:.1f}）**，最低为 **{asp_min_cat}（约 ¥{asp_min_v:.1f}）**。差距主要反映**价格带与产品结构**，不等于利润率（毛利数据未在本集中）。
- **尾部品类**：若某品类销量大但营收排名靠后，多为**低单价走量**；反之则可能**SKU 少但件单价高**。可结合 `charts/03_category.png` 与 `09_heatmap_category_month.png` 看是否存在「单一品类拖尾月份」。

### 3.2 产品表现

![Products](charts/04_products.png)

- **营收 Top 3**：{top_str}
- **营收 Bottom 3**：{bot_str}
- **Top 10 产品营收占全年**：约 **{top10_share:.1f}%**

### 3.2.1 数据解读与可能成因

- **头部集中**：Top10 营收合计约 **{top10_share:.1f}%**，说明**少数 SKU 贡献主要盘子**；运营上可优先保障头部 SKU 的库存与陈列，同时评估长尾是否过度挤占货架。
- **Bottom 产品**：低营收未必是「卖得差」，也可能是**定价低、上架时间短、或区域未铺货**；需回到原始门店/渠道维度才能下结论（本数据为汇总明细，未含门店维度）。

---

## 四、促销效果评估

![Promotion](charts/05_promotion.png)

### 4.1 促销类型对比

{md_table(['促销类型', '日均销量/件', '相比无促销提升', '营收贡献占比'], promo_rows)}

### 4.2 小结

以「无促销」日均销量为基准，**{best_promo_name}** 的相对提升最高（约 **+{best_promo_lift:.1f}%**）；无促销营收占比约 **{no_promo_rev_pct:.1f}%**。

### 4.3 数据解读与可能成因

- **「{best_promo_name}」为何在表观上最强**：在零售机制上，**明确底价+时间窗口**更容易触发冲动购买与比价完成；若数据里该类活动还叠加了**陈列/广告位**投入，则销量抬升是「价格+曝光」的联合结果，**不能单凭本表拆出纯价格效应**。
- **买赠 vs 打折**：若提升接近，通常体现为「感知折扣」相近；买赠对**客单价与毛利结构**的保护可能更好（仍缺成本字段，此处为经营常识推断）。
- **满减偏弱的可能原因**：门槛会**过滤小额冲动消费**；若门槛与客单分布不匹配，会出现「券看起来大、核销率低」。
- **无促销营收仍占 {no_promo_rev_pct:.1f}%**：说明并非「全靠促销活着」，但促销仍是**主要增量杠杆**；可进一步按品类拆分促销 ROI（需成本与费用数据）。

---

## 五、季节性规律

![Seasonality](charts/06_seasonality.png)

### 5.1 季节分布

{md_table(['季节', '营收（万元）', '占比'], se_rows)}

### 5.2 星期分布

工作日（周一至周五）SKU 日均销量约 **{mean_wk:.1f}** 件，周末约 **{mean_we:.1f}** 件；周末相对工作日约低 **{weekend_lower_pct:.1f}%**。

### 5.3 数据解读与可能成因

- **季节营收差异**：夏季/年末若占比更高，**常见解释**包括开学季文具、礼品消费、降温品类与节假日叠加；具体哪一类主导，需要按 **品类×月份** 热力图辅助判断（见 `charts/09_heatmap_category_month.png`）。
- **「周末低于工作日」**：在快消杂货业态里并不罕见，**可能机制**包括：样本对应门店位于**通勤型商圈**（工作日顺路购买）、周末客流被大型商场/线上渠道分流、或促销排期更多落在工作日。该结论依赖「记录口径 = 门店真实客流」的假设，**若数据含 ToB/团购且多在工作日结算，也会拉工作日均值**。

---

## 六、天气与节假日

![Weather & Holiday](charts/07_weather_holiday.png)

### 6.1 天气（按原始字段分组）

{md_table(['天气', '日均销量'], wx_rows)}

### 6.2 节假日

节假日 SKU 日均销量约 **{hol:.1f}** 件，非节假日约 **{nhol:.1f}** 件，相对提升约 **{hol_lift_pct:.1f}%**。

### 6.3 数据解读与可能成因

- **天气**：当前样本中，**{wx_hi_name}** 日均销量约 **{wx_hi_v:.1f}** 件，**{wx_lo_name}** 约 **{wx_lo_v:.1f}** 件，极差约 **{wx_rel_spread:.1f}%**。但天气与销量**高度容易混杂**：雨雪天可能同时伴随**促销加码、门店位置、节假日**等，本表未做多元回归或因果识别，**只能称「统计关联」**，不能写成「天气导致销量变化」。
- **节假日 +{hol_lift_pct:.1f}%**：节假日往往叠加**促销与出行消费**；本数据中节假日记录约占 **{hol_day_share:.1f}%** 行，若节假日同时是高促销期，则「节假日效应」与「促销效应」会**部分重叠**，解读时应避免重复归因。

---

## 七、价格档位与库存周转

![Price & Turnover](charts/08_price_turnover.png)

### 7.1 价格档位营收

{md_table(['价格档位', '营收（万元）', '占全年营收'], pl_rows)}

### 7.2 品类平均周转率

{md_table(['品类', '平均周转率'], turn_rows)}

### 7.3 数据解读与可能成因

- **价格带结构**：**高价**档营收约 **{float(hi_price['wan']):.0f} 万**（占 **{float(hi_price['pct']):.1f}%**），**低价**档约 **{float(lo_price['wan']):.0f} 万**（**{float(lo_price['pct']):.1f}%**）。若高价占比高，说明盘子更偏**高客单/组合购买**；若低价占比高则更偏**走量**。
- **周转**：**{turn_fast}** 平均周转率最高（约 **{turn_fast_v:.1f}%**），**{turn_slow}** 最低（约 **{turn_slow_v:.1f}%**）。周转高通常意味着**动销快或备货偏紧**，周转低可能是**高客单慢动销**或**备货偏多**；仍需结合安全库存与缺货率（本集未含）判断好坏。

---

## 八、数据校验（自动生成）

以下校验均针对本次写入报告所用的同一 `DataFrame`：

{check_lines}

---

## 九、使用说明

重新生成图表与本报告：在项目目录执行 `python3 analysis.py`。请勿手改本文件中的数字表格；**解读性文字**可在 `analysis.py` 的 `write_analysis_report` 函数中调整模板；如需更长论述可另建 `报告附录.md`。
"""
    path = os.path.join(out_dir, '分析报告.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f"Report written: {path}")


write_analysis_report(df, OUT, DATA)

print("\nAll done! 13 charts + SQL + 6 statistical tests + 分析报告.md (auto-generated).")
