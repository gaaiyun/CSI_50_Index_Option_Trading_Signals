@echo off
chcp 65001 >nul
title VolGuard Pro — 上证50期权风控雷达

echo.
echo  ╔══════════════════════════════════════════╗
echo  ║   VolGuard Pro  v6.0  启动检查          ║
echo  ╚══════════════════════════════════════════╝
echo.

REM 硬编码 Python 路径 (Python 3.12)
set PYTHON=C:\Users\gaaiy\AppData\Local\Programs\Python\Python312\python.exe
set STREAMLIT=C:\Users\gaaiy\AppData\Local\Programs\Python\Python312\Scripts\streamlit.exe

REM 检查 Python 可用性
"%PYTHON%" --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.12+
    pause
    exit /b 1
)

echo [1/4] 检查 Python 版本...
"%PYTHON%" --version

REM 检查核心依赖 (全量)
echo [2/4] 检查依赖包...
"%PYTHON%" -c "import streamlit, pandas, numpy, yfinance, akshare, arch, statsmodels, scipy, pyecharts, streamlit_echarts" 2>nul
if errorlevel 1 (
    echo [提示] 正在安装缺失依赖包，请稍候...
    "%PYTHON%" -m pip install streamlit pandas numpy yfinance akshare arch statsmodels scipy pyecharts streamlit-echarts -q
    if errorlevel 1 (
        echo [错误] 依赖安装失败，请检查网络连接后重试。
        pause
        exit /b 1
    )
    echo [完成] 依赖包已安装。
)

echo [3/4] 依赖校验通过。

echo [4/4] 启动看板...
echo.
echo  访问地址: http://localhost:8501
echo  按 Ctrl+C 停止服务
echo.

REM 切换到脚本所在目录，防止 DATA_DIR 路径漂移
cd /d "%~dp0"

"%STREAMLIT%" run "%~dp0app.py" --server.headless=true --browser.gatherUsageStats=false

pause
