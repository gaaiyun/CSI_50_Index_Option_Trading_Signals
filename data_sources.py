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


__all__ = ["fetch_50etf_options_sina"]

