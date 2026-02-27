# -*- coding: utf-8 -*-
"""VolGuard Pro — 数据源模块单元测试（含 mock 解析逻辑，在线测试标为 online）"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_sources import (
    fetch_50etf_options_sina,
    _get_option_months_sina,
    _get_option_codes_sina,
    _get_option_detail_sina,
)

REQUIRED_COLUMNS = [
    "代码", "名称", "最新价", "行权价", "涨跌幅",
    "成交量", "持仓量", "买入价", "卖出价",
]


class TestGetOptionMonthsSina:
    """_get_option_months_sina 解析逻辑"""

    def test_parses_json_months(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "result": {"data": {"contractMonth": ["2503", "2504", "2506"]}}
        }
        mock_resp.raise_for_status = MagicMock()
        with patch("data_sources.requests.get", return_value=mock_resp):
            months = _get_option_months_sina("510050")
        assert months == ["2503", "2504", "2506"]

    def test_filters_non_digit(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "result": {"data": {"contractMonth": ["2503", "xxx", "2506", ""]}}
        }
        mock_resp.raise_for_status = MagicMock()
        with patch("data_sources.requests.get", return_value=mock_resp):
            months = _get_option_months_sina("510050")
        assert months == ["2503", "2506"]

    def test_empty_on_request_failure(self):
        with patch("data_sources.requests.get", side_effect=Exception("timeout")):
            months = _get_option_months_sina("510050")
        assert months == []


class TestGetOptionCodesSina:
    """_get_option_codes_sina 解析 CON_OP_ 代码"""

    def test_extracts_call_and_put_codes(self):
        call_text = "var list=OP_UP_5100502503=\"CON_OP_10007301,CON_OP_10007302\";"
        put_text = "var list=OP_DOWN_5100502503=\"CON_OP_10007401\";"
        with patch("data_sources.requests.get") as mget:
            def side_effect(url, **kwargs):
                r = MagicMock()
                r.text = call_text if "OP_UP" in url else put_text
                r.encoding = "gbk"
                return r
            mget.side_effect = side_effect
            call_codes, put_codes = _get_option_codes_sina("510050", "2503")
        assert "CON_OP_10007301" in call_codes
        assert "CON_OP_10007302" in call_codes
        assert put_codes == ["CON_OP_10007401"]


class TestGetOptionDetailSina:
    """_get_option_detail_sina 解析 var hq_str_CON_OP_xxx 文本"""

    def test_parses_detail_line(self):
        # 模拟 42+ 字段的一行（索引 1 买价 2 最新价 3 卖价 5 持仓量 6 涨幅 7 行权价 36 合约简称 38/39 最高最低 40 成交量）
        # 需至少 42 字段: 索引 1 买价 2 最新价 3 卖价 5 持仓量 6 涨幅 7 行权价 36 合约简称 40 成交量
        parts = ["CON_OP_10007301", "0.149", "0.15", "0.151", "0", "45000", "1.2", "2.5", "0.148", "0.149"]
        parts += ["0"] * 26
        parts += ["50ETF购3月2500", "0.152", "0.148", "12000", "100000", "0"]
        value_str = ",".join(parts)
        line = f'var hq_str_CON_OP_10007301="{value_str}";'
        mock_resp = MagicMock()
        mock_resp.text = line
        mock_resp.encoding = "gbk"
        with patch("data_sources.requests.get", return_value=mock_resp):
            df = _get_option_detail_sina(["CON_OP_10007301"])
        assert len(df) == 1
        assert df.iloc[0]["合约代码"] == "CON_OP_10007301"
        assert df.iloc[0]["最新价"] == 0.15
        assert df.iloc[0]["行权价"] == 2.5
        assert df.iloc[0]["合约简称"] == "50ETF购3月2500"

    def test_empty_codes_returns_empty_df(self):
        df = _get_option_detail_sina([])
        assert df.empty


class TestFetch50etfOptionsSina:
    """fetch_50etf_options_sina 集成（mock 网络）"""

    def test_returns_tuple_df_msg(self):
        with patch("data_sources._get_option_months_sina", return_value=["2503"]), \
             patch("data_sources._get_option_codes_sina", return_value=(["CON_OP_1"], ["CON_OP_2"])), \
             patch("data_sources._get_option_detail_sina") as mdetail:
            call_df = pd.DataFrame({
                "合约代码": ["CON_OP_1"], "合约简称": ["50ETF购3月2500"],
                "最新价": [0.15], "行权价": [2.5], "涨幅": [1.0],
                "成交量": [100], "持仓量": [200], "买价": [0.14], "卖价": [0.16],
            })
            put_df = pd.DataFrame({
                "合约代码": ["CON_OP_2"], "合约简称": ["50ETF沽3月2500"],
                "最新价": [0.04], "行权价": [2.3], "涨幅": [0.5],
                "成交量": [50], "持仓量": [80], "买价": [0.039], "卖价": [0.041],
            })
            def side_detail(codes):
                if "CON_OP_1" in codes:
                    return call_df
                return put_df
            mdetail.side_effect = side_detail
            df, msg = fetch_50etf_options_sina()
        assert isinstance(df, pd.DataFrame)
        assert msg == "Sina options loaded"
        for col in REQUIRED_COLUMNS:
            assert col in df.columns, f"missing column {col}"
        assert len(df) >= 2

    def test_empty_months_returns_error_msg(self):
        with patch("data_sources._get_option_months_sina", return_value=[]):
            df, msg = fetch_50etf_options_sina()
        assert df.empty
        assert "Sina 合约月份获取失败" in msg or "失败" in msg


@pytest.mark.online
class TestFetch50etfOptionsSinaOnline:
    """真实 HTTP 请求，仅本地或显式 -m online 时运行"""

    def test_fetch_returns_columns_and_rows(self):
        df, msg = fetch_50etf_options_sina()
        assert isinstance(df, pd.DataFrame)
        assert isinstance(msg, str)
        if not df.empty:
            for col in REQUIRED_COLUMNS:
                assert col in df.columns
            assert len(df) > 0
