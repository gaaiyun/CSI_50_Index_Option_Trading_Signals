#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VolGuard Pro — PushPlus 消息推送模块 (v6.0)

安全规范: Token 和 Secret 不得硬编码.
使用方式:
  - 本地: 在项目根目录创建 .streamlit/secrets.toml, 写入:
        [pushplus]
        token = "your_token"
        secret = "your_secret"
  - Streamlit Cloud: 在 App Settings > Secrets 中配置
"""

import hashlib
import logging
import time
import requests
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class PushPlus:
    """PushPlus 消息推送客户端"""

    API_URL = "https://www.pushplus.plus/send"

    def __init__(self, token: str, secret: str = ""):
        if not token:
            raise ValueError("PushPlus token cannot be empty")
        self.token = token
        self.secret = secret

    def send(
        self,
        title: str,
        content: str,
        channel: str = "wechat",
        template: str = "markdown",
    ) -> Dict:
        """
        发送消息

        Args:
            title: 标题
            content: 正文 (支持 Markdown)
            channel: wechat / webhook / cp / mail
            template: html / txt / json / markdown
        """
        payload: Dict = {
            "token": self.token,
            "title": title,
            "content": content,
            "template": template,
            "channel": channel,
        }

        if self.secret:
            payload["timestamp"] = str(int(time.time() * 1000))
            payload["sign"] = self._generate_signature(payload["timestamp"])

        try:
            resp = requests.post(self.API_URL, json=payload, timeout=10)
            resp.raise_for_status()
            result = resp.json()
            return {
                "success": result.get("code") == 200,
                "code": result.get("code"),
                "msg": result.get("msg"),
                "data": result.get("data"),
            }
        except Exception as exc:
            logger.error(f"PushPlus send failed: {exc}")
            return {"success": False, "error": str(exc)}

    def _generate_signature(self, timestamp: str) -> str:
        """
        生成签名 — 正确顺序: timestamp + token (PushPlus API 规范)
        """
        message = f"{timestamp}{self.token}"
        if self.secret:
            message += self.secret
        return hashlib.sha256(message.encode("utf-8")).hexdigest()

    def send_signal(self, signal_type: str, message: str) -> Dict:
        """发送交易信号"""
        tags = {
            "建仓": "[建仓信号]",
            "止损": "[止损信号]",
            "平仓": "[平仓信号]",
            "警告": "[风险警告]",
        }
        title = f"50ETF期权策略 {tags.get(signal_type, signal_type)}"
        content = (
            f"## {signal_type}信号\n\n"
            f"**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"**内容**: {message}\n\n"
            f"---\n*VolGuard Pro 自动推送*"
        )
        return self.send(title, content)

    def send_alert(self, alert_type: str, content: str) -> Dict:
        """发送风险警告"""
        title = f"[{alert_type}] VolGuard Pro 期权监控"
        body = (
            f"## {alert_type}\n\n"
            f"**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"{content}\n\n"
            f"---\n*VolGuard Pro 自动推送*"
        )
        return self.send(title, body)


def get_push_client(token: Optional[str] = None, secret: Optional[str] = None) -> Optional[PushPlus]:
    """
    工厂函数: 从参数或环境变量创建推送客户端

    优先级: 参数 > st.secrets > 环境变量
    """
    import os
    _token = token or os.environ.get("PUSHPLUS_TOKEN", "")
    _secret = secret or os.environ.get("PUSHPLUS_SECRET", "")

    if not _token:
        logger.warning("PushPlus token not configured; push notifications disabled.")
        return None
    return PushPlus(_token, _secret)
