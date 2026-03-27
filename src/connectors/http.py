from __future__ import annotations

import requests
from typing import Optional

from src.config import settings


def get_json(url: str, params: Optional[dict] = None, timeout: int = 20) -> dict:
    headers = {"User-Agent": settings.http_user_agent}
    response = requests.get(url, params=params, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()
