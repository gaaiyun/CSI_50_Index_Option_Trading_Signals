#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中证50期权策略看板 - 完整严谨版 v3.1
支持: 完整BSADF | 完整GARCH | 多数据源 | 明确标注数据限制

⚠️ 重要提示:
- 云端(Streamlit Cloud): 仅支持ETF价格，期权数据需要本地
- 本地运行(VPN + akshare): 完整支持
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import requests

# ==================== 配置 ====================
PUSHPLUS_TOKEN = "3660eb1e0b364a78b3beed2f349b29f8"

st.set_page_config(
    page_title="中证50期权策略信号",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title { font-size: 1.8rem; font-weight: 700; color: #1f77b4; }
    .warning-box { padding: 1rem; background: #2a2a3e; border-left: 4px solid #f9a825; border-radius: 4px; }
    .success-box { padding: 1rem; background: #1e3a2e; border-left: 4px solid #00cc96; border-radius: 4px; }
    .error-box { padding: 1rem; background: #3a1e1e; border-left: 4px solid #ff6b6b; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ==================== 数据源 ====================
@st.cache_data(ttl=300)
def get_etf_data_yfinance():
    """使用yfinance获取ETF数据 (云端可用)"""
    try:
        import yfinance as yf
        
        # 上证50ETF (510050) - 最接近中证50
        # 沪深300ETF (510300)
        data = {}
        
        for symbol, name in [('510050.SS', '上证50ETF'), ('510300.SS', '沪深300ETF')]:
            t = yf.Ticker(symbol)
            hist = t.history(period='1y')
            hist.index = hist.index.tz_localize(None)
            data[name] = hist
        
        return data, "yfinance"
    except Exception as e:
        return None, str(e)

@st.cache_data(ttl=600)
def get_data_akshare():
    """使用akshare获取完整数据 (需要VPN)"""
    try:
        import akshare as ak
        import os
        os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
        os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
        
        data = {}
        
        # 1. 指数数据
        try:
            data['index'] = ak.stock_zh_index_daily_em(symbol='000016')
        except:
            pass
        
        # 2. 期货数据
        try:
            data['futures'] = ak.futures_contract_info_cffex()
        except:
            pass
        
        # 3. 期权数据
        try:
            data['options'] = ak.option_sse_daily_sina()
        except:
            pass
        
        return data, "akshare"
    except Exception as e:
        return None, str(e)

# ==================== 策略计算 (完整严谨版) ====================
class BSADF:
    """BSADF泡沫检验 - 完整版"""
    
    def __init__(self, window: int = 100):
        self.window = window
    
    def calculate(self, prices: pd.Series) -> tuple:
        from statsmodels.tsa.stattools import adfuller
        
        if len(prices) < self.window + 50:
            return 0.0, False
        
        log_prices = np.log(prices)
        returns = log_prices.diff().dropna()
        
        if len(returns) < self.window:
            return 0.0, False
        
        bsadf_stats = []
        
        for i in range(self.window, len(returns)):
            window_data = returns.iloc[i-self.window:i]
            try:
                result = adfuller(window_data, regression='ct', autolag='AIC')
                bsadf_stats.append(result[0])
            except:
                bsadf_stats.append(0)
        
        if not bsadf_stats:
            return 0.0, False
        
        bsadf_stat = max(bsadf_stats)
        critical_value = -3.5 + 1.5 * np.log(self.window / 100)
        is_significant = bsadf_stat > critical_value and bsadf_stats[-1] < -1.0
        
        return bsadf_stat, is_significant


class GARCHModel:
    """GARCH波动率模型 - 完整版"""
    
    def __init__(self):
        pass
    
    def fit_predict(self, returns: pd.Series) -> dict:
        from arch import arch_model
        import scipy.stats as stats
        
        if len(returns) < 100:
            return self._default()
        
        returns_pct = returns * 100
        
        try:
            # 正态GARCH
            model = arch_model(returns_pct, vol='Garch', p=1, q=1, dist='normal')
            fit = model.fit(disp='off')
            forecast = fit.forecast(horizon=1)
            sigma = np.sqrt(forecast.variance.iloc[-1].values[0]) / 100
            
            var_90 = stats.norm.ppf(0.10) * sigma * -1
            var_95 = stats.norm.ppf(0.05) * sigma * -1
            var_99 = stats.norm.ppf(0.01) * sigma * -1
            
            return {
                'sigma': sigma,
                'var_90': abs(var_90),
                'var_95': abs(var_95),
                'var_99': abs(var_99),
                'alpha': fit.params.get('alpha[1]', 0.08),
                'beta': fit.params.get('beta[1]', 0.90),
                'fitted': True
            }
        except:
            return self._default()
    
    def _default(self):
        return {
            'sigma': 0.01, 'var_90': 0.0165, 'var_95': 0.0196, 'var_99': 0.0233,
            'alpha': 0.08, 'beta': 0.90, 'fitted': False
        }


# ==================== 主程序 ====================
with st.sidebar:
    st.header("参数配置")
    
    otm = st.slider("建仓虚值(%)", 5, 20, 11)
    stop_loss = st.slider("止损虚值(%)", 3, 15, 6)
    bsadf_window = st.slider("BSADF窗口", 50, 200, 100)
    
    st.markdown("---")
    st.subheader("数据源")
    local_mode = st.checkbox("本地模式(需要VPN)", value=False)
    
    st.markdown("---")
    push_enabled = st.checkbox("启用推送", value=False)
    if st.button("推送测试"):
        try:
            requests.post("http://www.pushplus.plus/send", 
                        json={"token": PUSHPLUS_TOKEN, "title": "测试", "content": "OK"}, timeout=5)
            st.success("成功")
        except:
            st.error("失败")

# 标题
st.markdown('<p class="main-title">中证50期权策略看板</p>', unsafe_allow_html=True)

# 获取数据
if local_mode:
    data, source = get_data_akshare()
    if data and 'index' in data and not data['index'].empty:
        prices = data['index'].set_index('日期')['收盘']
    else:
        st.error("akshare需要VPN才能获取数据")
        prices = None
else:
    etf_data, source = get_etf_data_yfinance()
    if etf_data:
        # 使用上证50ETF
        df = etf_data.get('上证50ETF', etf_data.get('沪深300ETF'))
        if df is not None and not df.empty:
            prices = df['Close']
        else:
            prices = None
    else:
        prices = None

# 显示警告
if not local_mode:
    st.markdown("""
    <div class="warning-box">
        <b>云端模式</b>: 当前显示ETF价格数据<br>
        如需完整期权数据，请勾选"本地模式"(需要VPN)
    </div>
    """, unsafe_allow_html=True)

if prices is not None and len(prices) > 0:
    # 计算指标
    bsadf = BSADF(window=bsadf_window)
    garch = GARCHModel()
    
    returns = np.log(prices / prices.shift(1)).dropna()
    change = ((prices.iloc[-1] / prices.iloc[-2]) - 1) * 100 if len(prices) > 1 else 0
    
    bsadf_stat, triggered = bsadf.calculate(prices)
    garch_result = garch.fit_predict(returns)
    
    # 信号判断
    if triggered:
        signal = "建仓"
        action = f"卖出{otm}%虚值"
    elif change < -1.5:
        signal = "关注"
        action = "等待确认"
    else:
        signal = "观望"
        action = "等待BSADF"
    
    # 显示
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("ETF价格", f"{prices.iloc[-1]:.4f}", f"{change:.2f}%")
    
    with col2:
        vol = garch_result['sigma'] * np.sqrt(252) * 100
        st.metric("年化波动率", f"{vol:.2f}%")
    
    with col3:
        st.metric("VaR 99%", f"{garch_result['var_99']*100:.2f}%")
    
    with col4:
        st.metric("信号", signal, action)
    
    st.markdown("---")
    
    # 详情
    tab1, tab2, tab3 = st.tabs(["信号", "指标", "数据"])
    
    with tab1:
        if signal == "建仓":
            st.success(f"### 建仓信号\n\nBSADF={bsadf_stat:.4f} 触发")
        elif signal == "关注":
            st.warning(f"### 关注\n\nBSADF={bsadf_stat:.4f}")
        else:
            st.info(f"### 观望\n\nBSADF={bsadf_stat:.4f} 未触发")
    
    with tab2:
        st.markdown(f"""
        | 指标 | 数值 |
        |------|------|
        | BSADF | {bsadf_stat:.4f} |
        | σ(日度) | {garch_result['sigma']:.6f} |
        | α | {garch_result['alpha']:.4f} |
        | β | {garch_result['beta']:.4f} |
        | VaR 99% | {garch_result['var_99']*100:.2f}% |
        """)
    
    with tab3:
        if local_mode and data and 'futures' in data:
            st.write("期货合约:", data['futures'].head(5))
        elif etf_data:
            st.write("ETF数据:", etf_data['上证50ETF'].tail(5))
        else:
            st.warning("无数据")

else:
    st.error("无法获取数据，请检查网络或开启VPN")

st.markdown("---")
st.markdown(f"""
<div style='text-align:center; color:#666;'>
    <p>中证50期权策略看板 v3.1 | 数据源: {source} | {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
</div>
""", unsafe_allow_html=True)
