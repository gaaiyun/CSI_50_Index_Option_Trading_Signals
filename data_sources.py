#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VolGuard Pro — 期权数据源模块

主要解决问题：
- akshare.option_current_em 在海外环境 (如 Streamlit Cloud) 经常失效
- 本模块提供新浪财经数据源作为主数据源，akshare 作为可选备用

公开函数：
- fetch_50etf_options_sina() -> (df, source_msg)
"""

import logging
import re
from typing import List, Tuple

import pandas as pd
import requests
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

logger = logging.getLogger(__name__)


SINA_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0 Safari/537.36"
    ),
    "Referer": "https://stock.finance.sina.com.cn/",
}


def _get_option_months_sina(underlying: str = "510050") -> List[str]:
    """
    从新浪获取可用期权月份列表。

    返回类似 ["2503", "2504", "2506", ...] 的字符串列表。
    """
    url = (
        "http://stock.finance.sina.com.cn/futures/api/openapi.php/"
        "StockOptionService.getStockName"
    )
    try:
        resp = requests.get(url, headers=SINA_HEADERS, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        result = data.get("result", {}).get("data", {})
        # 这里不强依赖 cateList，只取 contractMonth
        months = result.get("contractMonth", []) or []
        # 保险起见做一次过滤
        months = [m for m in months if isinstance(m, str) and m.isdigit()]
        return months
    except Exception as e:
        logger.warning(f"Sina getStockName failed: {e}")
        return []


def _get_option_codes_sina(underlying: str, month: str) -> Tuple[List[str], List[str]]:
    """
    获取某个月份的认购 / 认沽合约代码列表。

    返回: (call_codes, put_codes)
    """
    call_url = f"http://hq.sinajs.cn/list=OP_UP_{underlying}{month}"
    put_url = f"http://hq.sinajs.cn/list=OP_DOWN_{underlying}{month}"

    call_codes: List[str] = []
    put_codes: List[str] = []

    try:
        call_resp = requests.get(call_url, headers=SINA_HEADERS, timeout=8)
        call_resp.encoding = "gbk"
        call_codes = re.findall(r"CON_OP_\d+", call_resp.text)
    except Exception as e:
        logger.warning(f"Sina call code list failed for {month}: {e}")

    try:
        put_resp = requests.get(put_url, headers=SINA_HEADERS, timeout=8)
        put_resp.encoding = "gbk"
        put_codes = re.findall(r"CON_OP_\d+", put_resp.text)
    except Exception as e:
        logger.warning(f"Sina put code list failed for {month}: {e}")

    return call_codes, put_codes


def _get_option_detail_sina(codes: List[str]) -> pd.DataFrame:
    """
    批量获取新浪期权合约行情。

    输出字段会在上层统一重命名为：
      - 代码 / 名称 / 最新价 / 行权价 / 涨跌幅 / 成交量 / 持仓量 / 买入价 / 卖出价
    """
    if not codes:
        return pd.DataFrame()

    code_str = ",".join(codes)
    url = f"http://hq.sinajs.cn/list={code_str}"

    try:
        resp = requests.get(url, headers=SINA_HEADERS, timeout=8)
        resp.encoding = "gbk"
    except Exception as e:
        logger.warning(f"Sina option detail request failed: {e}")
        return pd.DataFrame()

    rows = []
    for line in resp.text.strip().split("\n"):
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        code = key.replace("var hq_str_", "").strip()
        value = value.strip().strip('";')
        fields = value.split(",")

        # 期权返回通常有 42+ 字段，这里做长度保护
        if len(fields) < 40:
            continue

        try:
            rows.append(
                {
                    "合约代码": code,
                    "买价": float(fields[1]) if fields[1] else 0.0,
                    "最新价": float(fields[2]) if fields[2] else 0.0,
                    "卖价": float(fields[3]) if fields[3] else 0.0,
                    "持仓量": float(fields[5]) if fields[5] else 0.0,
                    "涨幅": float(fields[6]) if fields[6] else 0.0,
                    "行权价": float(fields[7]) if fields[7] else 0.0,
                    "昨收价": float(fields[8]) if fields[8] else 0.0,
                    "开盘价": float(fields[9]) if fields[9] else 0.0,
                    "合约简称": fields[36] if len(fields) > 36 else code,
                    "最高价": float(fields[38]) if fields[38] else 0.0,
                    "最低价": float(fields[39]) if fields[39] else 0.0,
                    "成交量": float(fields[40]) if len(fields) > 40 and fields[40] else 0.0,
                    "成交额": float(fields[41]) if len(fields) > 41 and fields[41] else 0.0,
                }
            )
        except Exception:
            # 单行解析失败直接跳过
            continue

    return pd.DataFrame(rows)


def fetch_50etf_options_sina() -> tuple[pd.DataFrame, str]:
    """
    从新浪财经获取 50ETF 全期权链数据。

    返回: (DataFrame, source_msg)
    DataFrame 至少包含 app.py 使用到的字段：
      - 代码 / 名称 / 最新价 / 行权价 / 涨跌幅 / 成交量 / 持仓量 / 买入价 / 卖出价
    """
    months = _get_option_months_sina("510050")
    if not months:
        return pd.DataFrame(), "Sina 合约月份获取失败"

    all_df: list[pd.DataFrame] = []

    for month in months:
        call_codes, put_codes = _get_option_codes_sina("510050", month)

        if call_codes:
            call_df = _get_option_detail_sina(call_codes)
            if not call_df.empty:
                call_df["类型"] = "认购"
                call_df["月份"] = month
                all_df.append(call_df)

        if put_codes:
            put_df = _get_option_detail_sina(put_codes)
            if not put_df.empty:
                put_df["类型"] = "认沽"
                put_df["月份"] = month
                all_df.append(put_df)

    if not all_df:
        return pd.DataFrame(), "Sina 期权行情为空"

    df = pd.concat(all_df, ignore_index=True)

    # 统一重命名为主程序习惯的中文列名
    rename_map = {
        "合约代码": "代码",
        "合约简称": "名称",
        "最新价": "最新价",
        "行权价": "行权价",
        "涨幅": "涨跌幅",
        "成交量": "成交量",
        "持仓量": "持仓量",
        "买价": "买入价",
        "卖价": "卖出价",
    }
    df = df.rename(columns=rename_map)

    return df, "Sina options loaded"


__all__ = ["fetch_50etf_options_sina", "add_implied_volatility"]


def _bs_call_price(S, K, T, r, sigma):
    """Black-Scholes 认购期权定价"""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def _bs_put_price(S, K, T, r, sigma):
    """Black-Scholes 认沽期权定价"""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _calculate_tte(name: str) -> float:
    """从合约名称提取到期时间（年）"""
    from datetime import datetime, timedelta
    try:
        if isinstance(name, str) and len(name) >= 11:
            year_month = name[7:11]
            year = 2000 + int(year_month[:2])
            month = int(year_month[2:])
            first_day = datetime(year, month, 1)
            first_wed = first_day + timedelta(days=(2 - first_day.weekday() + 7) % 7)
            expiry = first_wed + timedelta(weeks=3)
            days = (expiry - datetime.now()).days
            return max(1, days) / 365.0
    except Exception:
        pass
    return 30 / 365.0


def _implied_volatility(price, S, K, T, r, opt_type='call'):
    """从期权价格反推隐含波动率（Brent 方法）"""
    if price <= 0 or T <= 0:
        return 0.0
    
    intrinsic = max(S - K, 0) if opt_type == 'call' else max(K - S, 0)
    if price <= intrinsic:
        return 0.0
    
    bs_func = _bs_call_price if opt_type == 'call' else _bs_put_price
    
    try:
        def objective(sigma):
            return bs_func(S, K, T, r, sigma) - price
        
        iv = brentq(objective, 0.001, 5.0, xtol=1e-6, maxiter=100)
        return iv * 100  # 转换为百分比
    except Exception:
        return 20.0  # 默认 20%


def add_implied_volatility(df: pd.DataFrame, spot: float, risk_free_rate: float = 0.015) -> pd.DataFrame:
    """
    为期权数据添加隐含波动率列
    
    参数:
        df: 期权数据，需包含：名称、最新价、行权价
        spot: 标的现价
        risk_free_rate: 无风险利率
    
    返回:
        添加了"隐含波动率"列的 DataFrame
    """
    if df is None or df.empty:
        return df
    
    df = df.copy()
    
    # 如果已有隐含波动率且非空，直接返回
    if '隐含波动率' in df.columns and df['隐含波动率'].notna().any():
        return df
    
    ivs = []
    for _, row in df.iterrows():
        try:
            price = float(row.get('最新价', 0))
            strike = float(row.get('行权价', 0))
            name = str(row.get('名称', ''))
            
            if price <= 0 or strike <= 0:
                ivs.append(0.0)
                continue
            
            tte = _calculate_tte(name)
            opt_type = 'call' if '购' in name else 'put'
            
            iv = _implied_volatility(price, spot, strike, tte, risk_free_rate, opt_type)
            ivs.append(iv)
        except Exception:
            ivs.append(0.0)
    
    df['隐含波动率'] = ivs
    return df

