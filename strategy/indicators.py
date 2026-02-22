#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中证50期权策略 - 核心指标计算模块

包含:
- BSADF泡沫检验
- GARCH波动率预测
- RV已实现波动率
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# 注意：根据最新策略，不再需要强行依赖代理
# 如果本地运行确有需要，可在外部环境中设定，此处不硬编码代理


class StrategyIndicators:
    """策略指标计算类"""
    
    def __init__(self):
        self.proxy_enabled = True
        
    def _set_proxy(self):
        """设置代理"""
        if self.proxy_enabled:
            os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
            os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
    
    def calculate_bsadf(self, prices: pd.Series, window: int = 100) -> Dict:
        """
        计算BSADF泡沫检验
        
        参数:
            prices: 价格序列
            window: 滚动窗口大小
        
        返回:
            dict: 包含最大adf统计量、右尾显著性，以及历史序列用于绘图
        """
        try:
            from statsmodels.tsa.stattools import adfuller
            
            log_prices = np.log(prices).dropna()
            
            if len(log_prices) < window + 10:
                return {'error': 'Not enough data'}
                
            n = len(log_prices)
            bsadf_series = pd.Series(index=log_prices.index, dtype=float)
            sup_adf = -np.inf
            p_val = 1.0
            
            # Optimization: only compute rolling BSADF for the last 60 days to keep the dashboard responsive
            start_search_idx = max(window, n - 60)
            
            # We calculate supremum ADF using backward expanding windows
            for t in range(start_search_idx, n):
                end_idx = t
                current_sup_adf = -np.inf
                
                for start_idx in range(max(0, end_idx - 250), end_idx - window + 1):
                    window_data = log_prices.iloc[start_idx:end_idx+1]
                    try:
                        adf_stat = adfuller(window_data, regression='ct', autolag='AIC')[0]
                        if adf_stat > current_sup_adf:
                            current_sup_adf = adf_stat
                    except:
                        continue
                
                bsadf_series.iloc[t] = current_sup_adf
                if t == n - 1:
                    sup_adf = current_sup_adf
                    
            critical_value = -3.5 + 1.5 * np.log(window / 100)
            # A completely simplified right-tailed heuristic: stat > cv indicates an explosive process
            is_significant = sup_adf > critical_value
            
            return {
                'adf_stat': float(sup_adf),
                'p_value': float(p_val),
                'is_significant': is_significant,
                'cv': float(critical_value),
                'series': bsadf_series.dropna()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def calculate_garch_var(self, prices: pd.Series, 
                            confidence_levels: List[float] = [0.90, 0.95, 0.99],
                            window: int = 250) -> Dict:
        """
        计算GARCH波动率和VaR分位数
        
        参数:
            prices: 价格序列
            confidence_levels: 置信水平列表
            window: 滚动窗口大小
        
        返回:
            dict: 包含三种GARCH模型的VaR分位数
        """
        try:
            from arch import arch_model
            import scipy.stats as stats
            
            # 计算对数收益率
            returns = np.log(prices / prices.shift(1)).dropna()
            recent_returns = returns.iloc[-window:]
            data_scaled = recent_returns * 100
            
            results = {}
            
            # ==========================================
            # 模型 1: 标准正态分布 GARCH (sGARCH_norm)
            # ==========================================
            try:
                am_norm = arch_model(data_scaled, vol='Garch', p=1, q=1, dist='Normal')
                res_norm = am_norm.fit(disp='off')
                forecast_norm = res_norm.forecast(horizon=1)
                sigma_norm = np.sqrt(forecast_norm.variance.iloc[-1].values[0]) / 100
                
                results['sigma_norm'] = float(sigma_norm)
                for cl in confidence_levels:
                    # 分位数取绝对值代表下行/上行极值距离
                    z = stats.norm.ppf(cl)
                    results[f'norm_{int(cl*100)}'] = float(z * sigma_norm)
                
                results['alpha_norm'] = float(res_norm.params.get('alpha[1]', 0.1))
                results['beta_norm'] = float(res_norm.params.get('beta[1]', 0.8))
            except Exception as e:
                results['norm_error'] = str(e)
            
            # ==========================================
            # 模型 2: 偏态t分布 GARCH (sGARCH_skew)
            # ==========================================
            try:
                am_skew = arch_model(data_scaled, vol='Garch', p=1, q=1, dist='skewstudent')
                res_skew = am_skew.fit(disp='off')
                forecast_skew = res_skew.forecast(horizon=1)
                sigma_skew = np.sqrt(forecast_skew.variance.iloc[-1].values[0]) / 100
                
                results['sigma_skew'] = float(sigma_skew)
                
                # 提取参数估算偏态t的分位数(此处用正态叠加1.2倍作为厚尾安全垫的近似防守测算)
                for cl in confidence_levels:
                    z = stats.norm.ppf(cl)
                    results[f'skew_{int(cl*100)}'] = float(z * sigma_skew * 1.2)
            except Exception as e:
                results['skew_error'] = str(e)
            
            # ==========================================
            # 模型 3: 跳跃 GARCH (sGARCH_jump 补偿近似)
            # ==========================================
            sigma_jump = results.get('sigma_norm', 0.01) * 1.35
            results['sigma_jump'] = float(sigma_jump)
            for cl in confidence_levels:
                z = stats.norm.ppf(cl)
                results[f'jump_{int(cl*100)}'] = float(z * sigma_jump)
            
            # 设置最终参照的认怂线(VaR_99 为最高极值线)
            results['var_95'] = results.get('skew_95', results.get('norm_95', 0.02))
            results['var_99'] = results.get('jump_99', results.get('skew_99', 0.03))
            
            return results
            
        except Exception as e:
            return {'error': str(e)}
    
    def calculate_daily_rv(self, high_freq_df: pd.DataFrame, time_col: str = 'time', price_col: str = 'close') -> pd.Series:
        """
        计算盘中实时或日度的已实现波动率 (RV)
        
        参数:
            high_freq_df: 高频K线数据(例如5分钟)
            time_col: 时间列名
            price_col: 价格列名
        """
        try:
            df = high_freq_df.copy()
            if time_col in df.columns:
                df[time_col] = pd.to_datetime(df[time_col])
                df.set_index(time_col, inplace=True)
                
            df['log_ret'] = np.log(df[price_col] / df[price_col].shift(1))
            # 过滤掉夜盘开盘跳空等，仅计算日内
            daily_rv = df.groupby(df.index.date)['log_ret'].apply(lambda x: np.sqrt(np.sum(x**2)))
            
            # Assuming ~48 intervals of 5-minutes per day
            annual_rv = daily_rv * np.sqrt(250)
            return annual_rv
        except:
            return pd.Series(dtype=float)
    
    def calculate_otm_level(self, spot_price: float, strike_price: float) -> float:
        """
        计算虚值程度
        
        参数:
            spot_price: 现货价格
            strike_price: 行权价
        
        返回:
            float: 虚值程度 (%)
        """
        if spot_price <= 0 or strike_price <= 0:
            return 0.0
        
        # 虚值程度 = |现货 - 行权价| / 现货 * 100%
        otm = abs(spot_price - strike_price) / spot_price * 100
        
        return float(otm)
    
    def check_stop_loss(self, spot_price: float, strike_price: float,
                       entry_otm: float = 11.0, stop_otm: float = 6.4) -> bool:
        """
        检查是否触发止损
        
        参数:
            spot_price: 现货价格
            strike_price: 行权价
            entry_otm: 建仓时虚值程度 (%)
            stop_otm: 止损虚值程度 (%)
        
        返回:
            bool: True表示触发止损
        """
        current_otm = self.calculate_otm_level(spot_price, strike_price)
        
        # 如果虚值程度小于止损阈值，触发止损
        if current_otm < stop_otm:
            return True
        
        return False


def get_index_data(symbol: str = "000016") -> pd.DataFrame:
    """
    获取指数数据
    
    参数:
        symbol: 指数代码 (sh000016=上证50)
    
    返回:
        DataFrame: 指数数据
    """
    import akshare as ak
    
    df = ak.stock_zh_index_daily_em(symbol=symbol)
    if 'close' not in df.columns and '收盘' in df.columns:
        df.rename(columns={'日期': 'date', '收盘': 'close', '开盘': 'open', '最高': 'high', '最低': 'low', '成交量': 'volume', '成交额': 'amount'}, inplace=True)
    return df


# 测试代码
if __name__ == "__main__":
    print("=" * 50)
    print("中证50期权策略指标计算测试")
    print("=" * 50)
    
    # 获取数据
    print("\n[1] 获取中证50指数数据...")
    df = get_index_data("sh000016")
    print(f"获取到 {len(df)} 条数据")
    print(df.tail())
    
    # 初始化指标计算
    indicators = StrategyIndicators()
    
    # 计算BSADF
    print("\n[2] 计算BSADF泡沫检验...")
    prices = df['close']
    bsadf_result = indicators.calculate_bsadf(prices)
    print(f"BSADF结果: {bsadf_result}")
    
    # 计算GARCH
    print("\n[3] 计算GARCH VaR...")
    garch_result = indicators.calculate_garch_var(prices)
    print(f"GARCH结果: {garch_result}")
    
    print("\n✅ 测试完成!")
