from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq


@dataclass(frozen=True)
class GroqSettings:
    api_key: str
    model_name: str = "openai/gpt-oss-120b"
    temperature: float = 0.1
    max_tokens: int = 10000
    request_timeout: float = 60.0
    max_retries: int = 2

    @classmethod
    def from_env(cls) -> "GroqSettings":
        current_file = Path(__file__).resolve()
        repo_root = current_file.parents[1]

        # Load env values from project root and from config folder.
        load_dotenv(repo_root / ".env", override=False)
        load_dotenv(current_file.parent / ".env", override=False)

        # Fallback: leer st.secrets si estamos en Streamlit y falta alguna key
        try:
            import streamlit as st
            for key in ["GROQ_API_KEY", "GROQ_MODEL"]:
                if not os.environ.get(key):
                    val = st.secrets.get(key)
                    if val:
                        os.environ[key] = str(val)
        except Exception:
            pass

        api_key = os.getenv("GROQ_API_KEY", "").strip()
        model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()
        temperature = float(os.getenv("GROQ_TEMPERATURE", "0.1"))
        max_tokens = int(os.getenv("GROQ_MAX_TOKENS", "10000"))
        request_timeout = float(os.getenv("GROQ_REQUEST_TIMEOUT", "60"))
        max_retries = int(os.getenv("GROQ_MAX_RETRIES", "2"))

        return cls(
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            request_timeout=request_timeout,
            max_retries=max_retries,
        )


def build_groq_llm(settings: GroqSettings | None = None) -> ChatGroq:
    cfg = settings or GroqSettings.from_env()
    if not cfg.api_key:
        raise ValueError("GROQ_API_KEY is not configured.")

    return ChatGroq(
        model_name=cfg.model_name,
        groq_api_key=cfg.api_key,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        request_timeout=cfg.request_timeout,
        max_retries=cfg.max_retries,
    )
