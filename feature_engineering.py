#!/usr/bin/env python3
"""MINISO 2023 — Feature Engineering Pipeline

从原始 Kaggle CSV 生成分析就绪的特征数据集。每个步骤都保留了业务逻辑注释，
方便招聘者理解特征工程的思考过程。

Usage:
    python feature_engineering.py [--input raw.csv] [--output features.csv]

原始数据 → (本脚本) → miniso_sales_data_features.csv → 方案一二三分析
"""

import pandas as pd
import numpy as np
import argparse, os, sys
from sklearn.preprocessing import LabelEncoder

# ── 默认路径（支持命令行覆盖）──
DEFAULT_INPUT = os.path.expanduser('~/miniso_sales_data.csv')
DEFAULT_OUTPUT = os.path.expanduser('~/miniso_sales_data_features.csv')


def load_and_sort(path):
    """加载原始 CSV，解析日期，按产品+日期排序"""
    df = pd.read_csv(path, parse_dates=['date'])
    df = df.sort_values(['product', 'date']).reset_index(drop=True)
    print(f"Loaded {len(df)} rows, {df['product'].nunique()} products, "
          f"{df['date'].min().date()} ~ {df['date'].max().date()}")
    return df


def add_temporal_features(df):
    """提取日期衍生特征

    WHY: 零售销量有强周期性和季节性——
    - day_of_year 捕捉年度周期性波动
    - week_of_year 捕捉周级别趋势
    - quarter 捕捉季度财报周期
    """
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['date'].dt.quarter
    print(f"Added temporal features: day_of_year, week_of_year, quarter")
    return df


def encode_categoricals(df):
    """对类别特征做 Label Encoding

    WHY: 树模型（RF/XGBoost）可以直接处理数值编码的类别特征。
    使用 LabelEncoder 而非 One-Hot 可以避免高基数类别
    （如 product 有 24 个值）导致的维度膨胀。
    """
    cat_cols = ['price_level', 'weather', 'season', 'promotion_type']
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"  {col}: {list(le.classes_)} → {list(range(len(le.classes_)))}")
    return df, encoders


def add_lag_features(df):
    """按产品创建滞后特征（Lag Features）

    WHY: 销量预测的核心假设是「过去销量预示未来」。
    - lag_1:  捕捉日级别惯性（今天销量通常接近昨天）
    - lag_7:  捕捉周度模式（上周同一天可能类似）
    - lag_30: 捕捉月度趋势（大方向判断）

    每个产品独立计算，避免跨产品信息泄露。
    """
    for product in df['product'].unique():
        mask = df['product'] == product
        idx = df.loc[mask].index
        df.loc[idx, 'sales_lag_1'] = df.loc[idx, 'sales'].shift(1)
        df.loc[idx, 'sales_lag_7'] = df.loc[idx, 'sales'].shift(7)
        df.loc[idx, 'sales_lag_30'] = df.loc[idx, 'sales'].shift(30)
    print("Added lag features: sales_lag_1, sales_lag_7, sales_lag_30")
    return df


def add_rolling_features(df):
    """按产品创建滚动统计特征

    WHY: 滚动均值和标准差能捕捉短期趋势和波动性——
    - 7d_mean: 近期日均销量水平（短期趋势）
    - 30d_mean: 月均销量基线（中长期趋势）
    - 7d_std:  销量波动性（波动大的产品预测难度高，模型需要这个信息）

    min_periods=1 确保产品初期也能计算出值（用已有数据）。
    """
    for product in df['product'].unique():
        mask = df['product'] == product
        idx = df.loc[mask].index
        sales_series = df.loc[idx, 'sales']
        df.loc[idx, 'sales_7d_mean'] = sales_series.rolling(7, min_periods=1).mean().values
        df.loc[idx, 'sales_7d_std'] = sales_series.rolling(7, min_periods=1).std().values
        df.loc[idx, 'sales_30d_mean'] = sales_series.rolling(30, min_periods=1).mean().values
    print("Added rolling features: sales_7d_mean, sales_7d_std, sales_30d_mean")
    return df


def add_interaction_features(df):
    """创建交互特征

    WHY: 单一特征无法捕捉组合效应——
    - price_promotion_interaction: 同一促销在不同价格带效果不同
      （例如「打折」对高价品效果好，对低价品效果差）
    - season_weather_interaction: 同一季节不同天气影响不同
      （例如夏季晴天 vs 夏季雨天销量模式差异大）

    用编码值的乘积近似交互效应，适合树模型捕获非线性关系。
    """
    df['price_promotion_interaction'] = (
        df['price_level_encoded'] * df['promotion_type_encoded']
    )
    df['season_weather_interaction'] = (
        df['season_encoded'] * df['weather_encoded']
    )
    print("Added interaction features: price×promotion, season×weather")
    return df


def validate(df):
    """输出特征摘要，供人工检查"""
    print(f"\n{'='*60}")
    print("Feature Engineering Summary")
    print(f"{'='*60}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"New features added: "
          f"day_of_year, week_of_year, quarter, "
          f"sales_lag_1, sales_lag_7, sales_lag_30, "
          f"sales_7d_mean, sales_7d_std, sales_30d_mean, "
          f"price_level_encoded, weather_encoded, season_encoded, promotion_type_encoded, "
          f"price_promotion_interaction, season_weather_interaction")

    # 检查缺失值
    nan_cols = df.isna().sum()
    nan_cols = nan_cols[nan_cols > 0]
    if len(nan_cols) > 0:
        print(f"\nMissing values (expected in lag features for early dates):")
        for col, cnt in nan_cols.items():
            print(f"  {col}: {cnt} NaN ({cnt/len(df)*100:.1f}%)")
    else:
        print("\nNo missing values.")

    # 特征数量
    raw_cols = ['date', 'category', 'product', 'price_level', 'base_price',
                'sales', 'sales_amount', 'inventory', 'turnover_rate',
                'promotion_type', 'weather', 'season', 'is_holiday',
                'weekday', 'month', 'year']
    new_cols = [c for c in df.columns if c not in raw_cols]
    print(f"\nRaw columns: {len(raw_cols)}")
    print(f"Engineered columns: {len(new_cols)}")
    print(f"Total: {len(df.columns)}")


def main():
    parser = argparse.ArgumentParser(
        description='MINISO Feature Engineering Pipeline')
    parser.add_argument('--input', default=DEFAULT_INPUT,
                        help='Path to raw Kaggle CSV')
    parser.add_argument('--output', default=DEFAULT_OUTPUT,
                        help='Output path for features CSV')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        print(f"Download the dataset from Kaggle and place it at {args.input}")
        print(f"Or use: python feature_engineering.py --input /path/to/raw.csv")
        sys.exit(1)

    # ── Pipeline ──
    print(f"{'='*60}")
    print("MINISO Feature Engineering Pipeline")
    print(f"{'='*60}\n")

    df = load_and_sort(args.input)
    df = add_temporal_features(df)
    df, encoders = encode_categoricals(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_interaction_features(df)

    validate(df)

    # ── Save ──
    df.to_csv(args.output, index=False)
    print(f"\nSaved: {args.output}")
    print("Ready for analysis → prediction_model.py / price_elasticity_analysis.py")


if __name__ == '__main__':
    main()
