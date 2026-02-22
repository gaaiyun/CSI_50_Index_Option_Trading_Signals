# 上证50ETF期权 尾部风控高级策略看板 (v4.0)

## 一、项目概述

这是一套基于纯量化统计与极严苛尾部风控精神设计的量化大屏。其底层交易哲学是：**靠高胜率做空波动率(Short Volatility)，用立体化防线防止“千日砍柴一日烧”。**

**技术升级亮点**:
- **高级 Trading Echarts 看板**：可拖拽 K线缩放，直观观察历史 BSADF 泡沫区与指数关系。
- **免 VPN 云端兼顾**：通过 yfinance 直连主心骨日线，akshare 获取即时高频数据。
- **三重 GARCH 防护墙**：正态分布基础层 + 偏态厚尾分布(Skew-T) + 泊松跳跃补偿层，每天推演 T+1 极限在险价值(VaR)。
- **实时竞价合约测算**：即时读取东财/新浪期权链进行 T型比对，找出“最安全的高胜率靶心合约”。

---

## 二、配置信息汇总

### 2.1 PushPlus推送

| 项目 | 值 |
|------|-----|
| Token | `3660eb1e0b364a78b3beed2f349b29f8` |
| Secret | `ddff31dda80446cc878c163b2410bc5b` |
| 发送方式 | POST请求 |
| 模板 | markdown |

### 2.2 GitHub仓库

| 项目 | 值 |
|------|-----|
| 仓库地址 | `https://github.com/gaaiyun/CSI_50_Index_Option_Trading_Signals` |
| 分支 | main |
| 本地路径 | `C:\Users\gaaiy\Desktop\CSI_50_Index_Option_Trading_Signals` |

### 2.3 Streamlit Cloud

| 项目 | 值 |
|------|-----|
| 部署地址 | https://share.streamlit.io |
| App URL | (部署后获得) |

---

## 三、本地部署与演示

### 3.1 环境依赖

安装最新的依赖包以激活 ECharts，此系统在 Windows/Mac 上均推荐使用 Python 3.9+ 环境：

```bash
# 安装基础量化和看板环境
pip install streamlit pandas numpy akshare arch statsmodels scipy requests yfinance

# 安装高级图表环境
pip install pyecharts streamlit-echarts

# 进入项目运行
cd C:\Users\gaaiy\Desktop\CSI_50_Index_Option_Trading_Signals
streamlit run app.py
```

*注意：不再需要设置代理或开启任何 VPN 即可完整通过接口调用。*

### 3.3 浏览器访问

本地访问: http://localhost:8501

---

## 四、修改代码流程

### 4.1 修改代码

1. 用文本编辑器打开 `C:\Users\gaaiy\Desktop\CSI_50_Index_Option_Trading_Signals\app.py`
2. 修改代码
3. 保存

### 4.2 推送到GitHub

```bash
# 打开终端，进入项目目录
cd C:\Users\gaaiy\Desktop\CSI_50_Index_Option_Trading_Signals

# 添加所有修改
git add .

# 提交修改
git commit -m "更新说明"

# 推送到GitHub
git push
```

### 4.3 自动部署

推送到GitHub后，Streamlit Cloud会自动重新部署（等待1-2分钟）。

---

## 五、文件说明

| 文件 | 说明 |
|------|------|
| app.py | 主程序看板 |
| push_client.py | 推送模块(备用) |
| strategy/indicators.py | 指标计算 |
| README.md | 基础说明 |
| DEVELOPER.md | 开发文档 |

---

## 六、高阶量化模型核心表述

### 6.1 建仓防线 - 极值检验 (BSADF)
*   **思想来源**：寻找标的资产价格(上证50ETF)偏离随机游走的“爆炸性单位根”形态。
*   **计算机制**：提取过去500个交易日数据，滚动 100 天内滑动起点计算所有可能子区间的 ADF 统计量，提取在 Right-Tail 分布下的最大值。
*   **触发指令**：当最新日的 BSADF 统计量越过 `临界衰竭线` 时，认为泡沫破裂在即或单边极端行情结束，发出“卖出期权”的高胜率进场提示。

### 6.2 离场与风控 - 绝对的克制防线
*   **静态防护 (GARCH-VaR)**：
    每天计算 T+1 日极度悲观条件下的 95%、99% 在险价值幅度。如果卖出的虚值(OTM)程度（如原本偏离11%）因为现价波动，降至了 95% 认怂线以内内（例如降到离现价不足6.4%），代表防护壁被打破，**无条件平仓认输。绝不扛单。**
*   **流动风控 (高频RV监控)**：
    利用日内 K 线叠加盘中真实成交，若 5分钟 级别的年化 Realized Volatility 爆表，不等日线收位，盘中立刻出逃。

---

## 七、常见问题

### Q1: 看板K线无图、期权列表不显示？
- 请确保已安装新依赖：`pip install pyecharts streamlit-echarts`
- 交易所非交易时间段（如周末、收市）可能获取不到实时买卖盘，系统会给出友好提示降级展示底层数据。

### Q2: GitHub自动化无法上传？
- 建议将 `gaaiyun` GitHub 长期 Token 配置到本地 `credential.helper`。

---

## 八、联系方式

- GitHub: gaaiyun
- 项目: CSI_50_Index_Option_Trading_Signals

---

*更新时间: 2026-02-22*
*版本: v2.2*
