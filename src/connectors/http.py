from __future__ import annotations

import requests
from typing import Optional

from src.config import settings


def get_json(
    url: str,
    params: Optional[dict] = None,
    timeout: int = 20,
    extra_headers: Optional[dict] = None,
) -> dict:
    headers = {"User-Agent": settings.http_user_agent}
    if extra_headers:
        headers.update(extra_headers)
    response = requests.get(url, params=params, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()
