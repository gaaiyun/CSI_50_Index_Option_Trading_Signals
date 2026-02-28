#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试增强功能：反爬虫策略、东方财富数据源、数据清洗"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data_sources import (
    fetch_50etf_options_sina,
    fetch_50etf_options_eastmoney,
    clean_option_data,
)


def test_sina_with_retry():
    """测试 Sina 数据源（带重试机制）"""
    print("\n=== 测试 Sina 数据源（带重试和随机 UA）===")
    start = time.time()
    try:
        df, msg = fetch_50etf_options_sina()
        elapsed = time.time() - start
        
        if not df.empty:
            print(f"SUCCESS: 获取 {len(df)} 条数据")
            print(f"数据源: {msg}")
            print(f"列名: {list(df.columns)}")
            print(f"耗时: {elapsed:.2f}s")
            return True
        else:
            print(f"FAIL: {msg}")
            print(f"耗时: {elapsed:.2f}s")
            return False
    except Exception as e:
        elapsed = time.time() - start
        print(f"ERROR: {e}")
        print(f"耗时: {elapsed:.2f}s")
        return False


def test_eastmoney():
    """测试东方财富数据源"""
    print("\n=== 测试东方财富数据源 ===")
    start = time.time()
    try:
        df, msg = fetch_50etf_options_eastmoney()
        elapsed = time.time() - start
        
        if not df.empty:
            print(f"SUCCESS: 获取 {len(df)} 条数据")
            print(f"数据源: {msg}")
            print(f"列名: {list(df.columns)}")
            if "类型" in df.columns:
                print(f"认购/认沽分布: {df['类型'].value_counts().to_dict()}")
            print(f"耗时: {elapsed:.2f}s")
            return True, df
        else:
            print(f"INFO: {msg}")
            print(f"耗时: {elapsed:.2f}s")
            return False, None
    except Exception as e:
        elapsed = time.time() - start
        print(f"ERROR: {e}")
        print(f"耗时: {elapsed:.2f}s")
        return False, None


def test_data_cleaning(df):
    """测试数据清洗功能"""
    print("\n=== 测试数据清洗功能 ===")
    if df is None or df.empty:
        print("SKIP: 无数据可清洗")
        return False
    
    try:
        original_len = len(df)
        cleaned_df = clean_option_data(df)
        cleaned_len = len(cleaned_df)
        
        print(f"SUCCESS: 清洗完成")
        print(f"原始数据: {original_len} 条")
        print(f"清洗后: {cleaned_len} 条")
        print(f"移除: {original_len - cleaned_len} 条")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def main():
    print("=" * 60)
    print("VolGuard Pro - 增强功能测试")
    print("=" * 60)
    
    results = []
    
    # 测试 Sina（带重试）
    results.append(("Sina 数据源（重试）", test_sina_with_retry()))
    
    # 测试东方财富
    success, df = test_eastmoney()
    results.append(("东方财富数据源", success))
    
    # 测试数据清洗
    if df is not None:
        results.append(("数据清洗", test_data_cleaning(df)))
    
    # 汇总
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{name}: {status}")
    
    passed_count = sum(1 for _, p in results if p)
    print(f"\n通过: {passed_count}/{len(results)}")


if __name__ == "__main__":
    main()
