# -*- coding: utf-8 -*-
"""VolGuard Pro — 主应用轻量单元测试（逻辑/缓存，不跑完整 Streamlit UI）"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_import_app_no_error():
    """导入 app 模块不抛异常"""
    import app
    assert app is not None


def test_get_etf_510050_returns_loaded_when_cache_has_data():
    """缓存有数据时 get_etf_510050 返回 (df, 'loaded')"""
    import app as app_mod
    dummy_df = pd.DataFrame({"Close": [2.5, 2.51], "Date": pd.date_range("2024-01-01", periods=2)})
    with patch.object(app_mod, "_cached_etf_fetch", return_value=dummy_df):
        with patch.object(app_mod, "load_local_cache", return_value=None):
            df, msg = app_mod.get_etf_510050(force_refresh=False)
    assert msg == "loaded"
    assert df is not None and not df.empty


def test_get_etf_510050_returns_no_data_when_cache_empty():
    """缓存为空且后端无数据时返回 (None, 'no data')"""
    import app as app_mod
    with patch.object(app_mod, "_cached_etf_fetch", return_value=None):
        with patch.object(app_mod, "load_local_cache", return_value=None):
            df, msg = app_mod.get_etf_510050(force_refresh=False)
    assert df is None
    assert msg == "no data"


def test_get_options_data_returns_loaded_when_cache_has_data():
    """期权缓存有数据时 get_options_data 返回 (df, 'loaded')"""
    import app as app_mod
    dummy_opt = pd.DataFrame({
        "代码": ["CON_OP_1"], "名称": ["50ETF购3月2500"], "最新价": [0.15],
        "行权价": [2.5], "涨跌幅": [0], "成交量": [100], "持仓量": [200],
        "买入价": [0.14], "卖出价": [0.16],
    })
    with patch.object(app_mod, "_cached_options_fetch", return_value=(dummy_opt, "Cache loaded")):
        with patch.object(app_mod, "load_local_cache", return_value=None):
            df, msg = app_mod.get_options_data(force_refresh=False)
    assert msg == "loaded"
    assert df is not None and not df.empty


def test_get_options_data_message_contains_error_when_backend_fails():
    """期权后端失败时返回 (None, msg)，msg 含错误提示"""
    import app as app_mod
    with patch.object(app_mod, "_cached_options_fetch", return_value=(None, "Sina 合约月份获取失败")):
        with patch.object(app_mod, "load_local_cache", return_value=None):
            df, msg = app_mod.get_options_data(force_refresh=False)
    assert df is None
    assert "Sina" in msg or "失败" in msg or "Akshare" in msg
