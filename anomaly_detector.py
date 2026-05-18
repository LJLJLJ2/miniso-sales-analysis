#!/usr/bin/env python3
"""
自动异常检测脚本 — crontab 定时运行
每天自动检查昨日销量是否异常（偏高/偏低），发现异常写入日志+告警记录
"""

import pandas as pd
import numpy as np
import os, sys, json, warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# ── 配置 ──
DATA_PATH = '/Users/zhoujingjing/miniso_sales_data.csv'
LOG_DIR = '/Users/zhoujingjing/Desktop/miniso_sales_analysis'
LOG_FILE = f'{LOG_DIR}/anomaly_check.log'
ALERT_FILE = f'{LOG_DIR}/anomaly_alerts.json'
THRESHOLD_STD = 2.0  # 超过 2 个标准差视为异常

# ── 辅助函数 ──
def log(msg, level='INFO'):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [{level}] {msg}"
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')
    print(line)


def main():
    log("=" * 55)
    log("异常检测任务开始执行")
    log("=" * 55)

    # ── 1. 加载数据 ──
    if not os.path.exists(f'{LOG_DIR}/analysis.py'):
        log("项目目录不存在？脚本应在定时任务中自动运行", 'WARN')

    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    df = df.sort_values(['product', 'date']).reset_index(drop=True)

    # 取"昨天"（演示模式：用数据最后一天模拟）
    last_date = df['date'].max()
    yesterday = last_date
    log(f"检查日期: {yesterday.date()}（演示模式使用数据最后一天）")

    day_data = df[df['date'] == yesterday].copy()

    if len(day_data) == 0:
        log(f"{yesterday.date()} 无数据，跳过", 'WARN')
        return

    # ── 2. 计算每款产品的"预期销量" ──
    # 生产环境用模型预测，演示用产品近30天均值做基线
    baseline = df[(df['date'] < yesterday) & (df['date'] >= yesterday - timedelta(days=30))]
    baseline = baseline.groupby('product')['sales'].agg(['mean', 'std']).fillna(0)

    anomalies = []
    for _, row in day_data.iterrows():
        prod = row['product']
        actual = row['sales']

        if prod not in baseline.index:
            continue

        expected = baseline.loc[prod, 'mean']
        std = baseline.loc[prod, 'std']
        if pd.isna(expected) or expected == 0:
            continue

        deviation_pct = (actual - expected) / expected * 100

        # 判断异常：偏差超过 2 个标准差 OR 偏差率超过 ±50%
        is_anomaly = False
        reason = ''
        if std > 0 and abs(actual - expected) > THRESHOLD_STD * std:
            is_anomaly = True
            reason = f'超过 {THRESHOLD_STD}σ（μ={expected:.0f}, σ={std:.0f}）'
        elif abs(deviation_pct) > 50:
            is_anomaly = True
            reason = f'偏差 {deviation_pct:.1f}%（预期 {expected:.0f} 件）'

        if is_anomaly:
            direction = '⚠ 偏高' if actual > expected else '🔻 偏低'
            anomalies.append({
                '日期': str(yesterday.date()),
                '产品': prod,
                '品类': row['category'],
                '预期销量': round(expected, 1),
                '实际销量': int(actual),
                '偏差率': f'{deviation_pct:.1f}%',
                '方向': direction,
                '原因': reason
            })

    # ── 3. 输出结果 ──
    log(f"共检查 {len(day_data)} 款产品")

    if anomalies:
        log(f"发现 {len(anomalies)} 项异常:", 'ALERT')
        for a in anomalies:
            log(f"  {a['方向']:4} {a['产品']:6} | 预期 {a['预期销量']:>6.0f} 件 | "
                f"实际 {a['实际销量']:>5} 件 | 偏差率 {a['偏差率']:>8} | {a['原因']}")

        # 写入告警文件供外部读取
        existing = []
        if os.path.exists(ALERT_FILE):
            try:
                with open(ALERT_FILE, 'r') as f:
                    existing = json.load(f)
            except Exception:
                existing = []
        existing.extend(anomalies)
        with open(ALERT_FILE, 'w', encoding='utf-8') as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        log(f"告警已写入 {ALERT_FILE}")
    else:
        log("✅ 所有产品销售正常，无异常", 'OK')

    log("=" * 55)
    log("异常检测任务执行完毕")
    log("=" * 55)


if __name__ == '__main__':
    main()
