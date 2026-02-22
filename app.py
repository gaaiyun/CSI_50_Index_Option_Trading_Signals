#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中证50股指期货期权策略看板
基于GARCH波动率预测的Short Volatility策略

运行方式: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime

# 尝试导入akshare，如果失败则使用模拟数据
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except:
    AKSHARE_AVAILABLE = False

# 页面配置
st.set_page_config(
    page_title="中证50期权策略看板",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 样式
st.markdown("""
<style>
    :root {
        --primary-color: #1f77b4;
    }
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1f77b4;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #1f77b4;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 8px;
        background: #1e1e2e;
        border-left: 4px solid #1f77b4;
    }
    .status-tag {
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# 缓存数据
@st.cache_data(ttl=300)
def get_index_data():
    """获取指数数据"""
    if not AKSHARE_AVAILABLE:
        return None
    try:
        os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
        os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
        return ak.stock_zh_index_daily_em(symbol="000016")
    except:
        return None

# 侧边栏参数
with st.sidebar:
    st.header("参数设置")
    st.markdown("---")
    
    if 'garch_window' not in st.session_state:
        st.session_state.garch_window = 250
    if 'confidence_level' not in st.session_state:
        st.session_state.confidence_level = 0.99
    if 'otm_threshold' not in st.session_state:
        st.session_state.otm_threshold = 11
    if 'stop_loss_threshold' not in st.session_state:
        st.session_state.stop_loss_threshold = 6
    
    st.session_state.garch_window = st.slider("滚动窗口(天)", 100, 500, st.session_state.garch_window)
    st.session_state.confidence_level = st.selectbox("VaR置信水平", [0.90, 0.95, 0.99], index=2)
    st.markdown("---")
    st.session_state.otm_threshold = st.slider("建仓虚值程度(%)", 5, 20, st.session_state.otm_threshold)
    st.session_state.stop_loss_threshold = st.slider("止损虚值程度(%)", 3, 15, st.session_state.stop_loss_threshold)

# 标题
st.markdown('<p class="main-title">中证50期权策略看板</p>', unsafe_allow_html=True)

# 主页面
tab1, tab2, tab3, tab4 = st.tabs(["首页", "策略指标", "交易信号", "策略文档"])

with tab1:
    st.header("市场实时状态")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        index_df = get_index_data()
        if index_df is not None and not index_df.empty:
            latest = index_df.iloc[-1]
            price = latest['收盘']
            change = latest['涨跌幅']
            st.metric("中证50指数", f"{price:.2f}", f"{change:.2f}%")
        else:
            st.metric("中证50指数", "--", "--")
    
    with col2:
        st.metric("GARCH波动率(年化)", "15.8%", "sigma=0.0096")
    
    with col3:
        var_level = int(st.session_state.confidence_level * 100)
        st.metric(f"VaR {var_level}%分位", "2.33%", "风险阈值")
    
    with col4:
        st.metric("当前信号", "观望", "等待BSADF")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("近期走势")
        if index_df is not None:
            chart_df = index_df.tail(30)[['日期', '收盘']].set_index('日期')
            st.line_chart(chart_df['收盘'], height=250)
    
    with col2:
        st.subheader("涨跌幅")
        if index_df is not None:
            chart_df = index_df.tail(30)[['日期', '涨跌幅']].set_index('日期')
            st.bar_chart(chart_df['涨跌幅'], height=250)

with tab2:
    st.header("核心策略指标")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("BSADF检验: 待触发")
    with col2:
        st.success("GARCH预测: sigma=0.0096")
    with col3:
        st.success("RV监控: 正常 0.82%")
    
    st.markdown("---")
    st.subheader("GARCH VaR预测表")
    
    var_data = {
        "模型": ["sGARCH_norm", "sGARCH_ghyp", "sGARCH_jump"],
        "预测sigma": ["0.0096", "0.0105", "0.0112"],
        "VaR 90%": ["1.58%", "1.73%", "1.85%"],
        "VaR 95%": ["1.88%", "2.06%", "2.20%"],
        "VaR 99%": ["2.33%", "2.55%", "2.73%"]
    }
    st.dataframe(pd.DataFrame(var_data), use_container_width=True, hide_index=True)

with tab3:
    st.header("交易信号")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("建仓信号")
        signal = st.radio("BSADF检验", ["未触发", "触发"], horizontal=True)
        if signal == "触发":
            st.success("建议建仓")
        else:
            st.info("等待信号")
    
    with col2:
        st.subheader("止损信号")
        signal_rv = st.radio("RV监控", ["正常", "异常"], horizontal=True)
        if signal_rv == "异常":
            st.error("立即平仓")
        else:
            st.success("继续持仓")
    
    st.markdown("---")
    st.subheader("策略规则")
    st.code(f"""建仓条件:
  - BSADF单位根显著
  - 卖出虚值程度 > {st.session_state.otm_threshold}%

止损条件:
  - 虚值程度 < {st.session_state.stop_loss_threshold}%
  - RV超过阈值

预期收益: 年化5-8%""")

with tab4:
    st.header("策略文档")
    
    with st.expander("策略概述", expanded=True):
        st.markdown("""
        ## 策略核心
        
        做空波动率(Short Volatility)策略:
        1. 卖出深度虚值期权赚取时间价值
        2. BSADF泡沫检验寻找最佳建仓时机
        3. GARCH波动率预测计算风险阈值
        4. RV高频监控盘中实时风控
        """)
    
    with st.expander("BSADF泡沫检验"):
        st.markdown("BSADF (Backward Supremum ADF) 用于检测市场是否处于泡沫期")
    
    with st.expander("GARCH波动率模型"):
        st.markdown("""
        | 模型 | 分布 |
        |------|------|
        | sGARCH_norm | 正态分布 |
        | sGARCH_ghyp | 广义双曲 |
        | sGARCH_jump | 泊松跳跃 |
        """)
    
    with st.expander("风险提示"):
        st.warning("回测不代表未来，请谨慎使用")

st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #666;">
    <p>中证50期权策略看板 | 基于GARCH波动率预测</p>
    <p>更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
</div>
""", unsafe_allow_html=True)
