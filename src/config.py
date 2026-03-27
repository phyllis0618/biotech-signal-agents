from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    http_user_agent: str = os.getenv("HTTP_USER_AGENT", "biotech-signal-agents/0.1")


settings = Settings()
