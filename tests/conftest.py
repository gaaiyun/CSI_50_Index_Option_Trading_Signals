import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from strategy.indicators import StrategyIndicators


def pytest_configure(config):
    config.addinivalue_line("markers", "online: mark test as requiring real network (run with -m online)")


@pytest.fixture
def indicators():
    return StrategyIndicators()


@pytest.fixture
def sample_prices():
    """300 个交易日的模拟 ETF 价格（随机游走）"""
    np.random.seed(42)
    n = 300
    returns = np.random.normal(0.0003, 0.012, n)
    prices = 2.5 * np.cumprod(1 + returns)
    idx = pd.bdate_range("2024-01-02", periods=n)
    return pd.Series(prices, index=idx, name="Close")


@pytest.fixture
def sample_prices_short():
    """50 个数据点 — 用于测试数据不足的边界情况"""
    np.random.seed(99)
    prices = 2.5 * np.cumprod(1 + np.random.normal(0, 0.01, 50))
    idx = pd.bdate_range("2024-06-01", periods=50)
    return pd.Series(prices, index=idx, name="Close")


@pytest.fixture
def sample_options_df():
    """模拟期权链 DataFrame，包含 app.py 需要的全部列"""
    return pd.DataFrame(
        {
            "代码": [
                "CON_OP_10007301",
                "CON_OP_10007302",
                "CON_OP_10007303",
                "CON_OP_10007401",
                "CON_OP_10007402",
                "CON_OP_10007403",
            ],
            "名称": [
                "50ETF购3月2500",
                "50ETF购3月2600",
                "50ETF购3月2700",
                "50ETF沽3月2300",
                "50ETF沽3月2200",
                "50ETF沽3月2100",
            ],
            "最新价": [0.1500, 0.0800, 0.0300, 0.0400, 0.0150, 0.0050],
            "行权价": [2.500, 2.600, 2.700, 2.300, 2.200, 2.100],
            "涨跌幅": [1.2, -0.5, -2.1, 0.8, 1.5, -0.3],
            "成交量": [12000, 8500, 3200, 9500, 6000, 2100],
            "持仓量": [45000, 32000, 15000, 38000, 25000, 10000],
            "隐含波动率": [18.5, 19.2, 20.1, 17.8, 18.9, 21.5],
            "买入价": [0.1490, 0.0790, 0.0290, 0.0390, 0.0140, 0.0045],
            "卖出价": [0.1510, 0.0810, 0.0310, 0.0410, 0.0160, 0.0055],
        }
    )


@pytest.fixture
def sample_hf_df():
    """模拟 2 个交易日的 5 分钟 K 线"""
    np.random.seed(7)
    bars_per_day = 48
    n = bars_per_day * 2

    times = []
    for day in ["2024-06-03", "2024-06-04"]:
        base = pd.Timestamp(f"{day} 09:30")
        times.extend([base + pd.Timedelta(minutes=5 * i) for i in range(bars_per_day)])

    prices = 2.5 * np.cumprod(1 + np.random.normal(0, 0.002, n))
    return pd.DataFrame({"time": times, "close": prices})
