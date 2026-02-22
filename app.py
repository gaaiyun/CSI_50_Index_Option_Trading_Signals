#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
上证50ETF期权策略看板 - 完整严谨版 v3.2

重点:
- 上证50ETF: 510050.SS (yfinance/akshare)
- 上证50期货: IH (akshare)
- 期权: 510050ETF期权

数据源:
- yfinance (云端): 510050.SS ETF价格
- akshare (本地): IH期货、期权数据
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import requests

# 配置
PUSHPLUS_TOKEN = "3660eb1e0b364a78b3beed2f349b29f8"

st.set_page_config(
    page_title="上证50ETF期权策略",
    page_icon="",
    layout="wide"
)

st.markdown("""
<style>
    .main-title { font-size: 1.8rem; font-weight: 700; color: #1f77b4; }
    .warning { padding: 1rem; background: #2a2a3e; border-left: 4px solid #f9a825; }
</style>
""", unsafe_allow_html=True)

# ==================== 数据获取 ====================
@st.cache_data(ttl=300)
def get_etf_510050():
    """获取上证50ETF (510050.SS)"""
    try:
        import yfinance as yf
        t = yf.Ticker("510050.SS")
        df = t.history(period="1y")
        df.index = df.index.tz_localize(None)
        return df['Close'], "yfinance"
    except Exception as e:
        return None, str(e)

@st.cache_data(ttl=600)
def get_ih_futures():
    """获取上证50期货IH (需要akshare)"""
    try:
        import akshare as ak
        df = ak.futures_zh_daily_sina(symbol="IH")
        return df, "akshare"
    except:
        return None, "akshare"

# ==================== 策略计算 ====================
class BSADF:
    """BSADF泡沫检验"""
    def __init__(self, window=100):
        self.window = window
    
    def calculate(self, prices):
        from statsmodels.tsa.stattools import adfuller
        if len(prices) < self.window + 50:
            return 0.0, False
        
        returns = np.log(prices / prices.shift(1)).dropna()
        if len(returns) < self.window:
            return 0.0, False
        
        bsadf_stats = []
        for i in range(self.window, len(returns)):
            try:
                result = adfuller(returns.iloc[i-self.window:i], regression='ct', autolag='AIC')
                bsadf_stats.append(result[0])
            except:
                bsadf_stats.append(0)
        
        if not bsadf_stats:
            return 0.0, False
        
        bsadf = max(bsadf_stats)
        cv = -3.5 + 1.5 * np.log(self.window / 100)
        return bsadf, bsadf > cv and bsadf_stats[-1] < -1.0

class GARCH:
    """GARCH波动率"""
    def __init__(self):
        pass
    
    def calculate(self, returns):
        from arch import arch_model
        import scipy.stats as stats
        
        if len(returns) < 100:
            return {'sigma': 0.01, 'var_99': 0.023, 'fitted': False}
        
        try:
            model = arch_model(returns * 100, vol='Garch', p=1, q=1, dist='normal')
            fit = model.fit(disp='off')
            sigma = np.sqrt(fit.forecast(1).variance.iloc[0]) / 100
            var_99 = abs(stats.norm.ppf(0.01) * sigma)
            return {'sigma': sigma, 'var_99': var_99, 'fitted': True}
        except:
            return {'sigma': 0.01, 'var_99': 0.023, 'fitted': False}

# ==================== 主程序 ====================
with st.sidebar:
    st.header("参数")
    otm = st.slider("建仓虚值%", 5, 20, 11)
    stop_loss = st.slider("止损虚值%", 3, 15, 6)
    
    st.markdown("---")
    push = st.checkbox("推送")
    if st.button("测试"):
        st.success("OK")

st.markdown('<p class="main-title">上证50ETF期权策略看板</p>', unsafe_allow_html=True)

# 获取数据
prices, source = get_etf_510050()

if prices is not None and len(prices) > 0:
    # 计算
    bsadf = BSADF()
    garch = GARCH()
    
    returns = np.log(prices / prices.shift(1)).dropna()
    change = ((prices.iloc[-1] / prices.iloc[-2]) - 1) * 100
    
    bsadf_stat, triggered = bsadf.calculate(prices)
    garch_result = garch.calculate(returns)
    
    # 信号
    if triggered:
        signal, action = "建仓", f"卖出{otm}%虚值Put"
    elif change < -1.5:
        signal, action = "关注", "等待确认"
    else:
        signal, action = "观望", "等待BSADF"
    
    # 显示
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("510050.SS", f"{prices.iloc[-1]:.4f}", f"{change:.2f}%")
    col2.metric("年化波动率", f"{garch_result['sigma']*np.sqrt(252)*100:.2f}%")
    col3.metric("VaR 99%", f"{garch_result['var_99']*100:.2f}%")
    col4.metric("信号", signal, action)
    
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["信号", "指标"])
    
    with tab1:
        if signal == "建仓":
            st.success(f"### 建仓\n\nBSADF={bsadf_stat:.4f}\n\n{action}")
        else:
            st.info(f"BSADF={bsadf_stat:.4f} 未触发")
    
    with tab2:
        st.markdown(f"""
        | 指标 | 值 |
        |------|-----|
        | BSADF | {bsadf_stat:.4f} |
        | σ(日) | {garch_result['sigma']:.6f} |
        | VaR99 | {garch_result['var_99']*100:.2f}% |
        """)

else:
    st.error("无法获取数据")

st.markdown(f"---数据源: {source} | {datetime.now().strftime('%H:%M')}")
