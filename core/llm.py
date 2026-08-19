import os
from typing import Dict, List

import requests
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from core.config import get_secret, logger

class LLMManager:
    FALLBACK_MESSAGE = (
        "I'm sorry, I'm having temporary technical issues. "
        "Please try again in a moment."
    )

    def __init__(
        self,
        openrouter_model: str = None,
        gemini_model: str = "gemini-3.6-flash",
        temperature: float = 0.2,
        timeout: int = 10,
    ) -> None:
        self.openrouter_model = openrouter_model or os.getenv(
            "OPENROUTER_MODEL", "cohere/north-mini-code:free"
        )
        self.gemini_model = os.getenv("GEMINI_MODEL") or gemini_model
        self.gemini_key = get_secret("GEMINI_API_KEY")
        self.gemini_url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.gemini_model}:generateContent"
        )
        self.secondary_llm = ChatOpenAI(
            model=self.openrouter_model,
            openai_api_key=get_secret("OPEN_ROUTER_KEY") or "not-set",
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=temperature,
            max_retries=1,
            timeout=timeout,
        )
        self.usage_log: List[Dict[str, str]] = []

    def chat(self, system_prompt: str, user_prompt: str) -> Dict[str, str]:
        try:
            text = self._call_gemini(system_prompt, user_prompt)
            self.usage_log.append({"source": "gemini", "status": "success"})
            if len(self.usage_log) > 500:
                self.usage_log.pop(0)
            return {"content": text, "source": "gemini", "status": "success"}
        except Exception as e:
            logger.error(f"[LLMManager] Gemini failed: {e}")
            try:
                messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
                response = self.secondary_llm.invoke(messages)
                self.usage_log.append({"source": "openrouter_backup", "status": "fallback"})
                if len(self.usage_log) > 500:
                    self.usage_log.pop(0)
                return {"content": response.content, "source": "openrouter_backup", "status": "fallback"}
            except Exception as e2:
                logger.error(f"[LLMManager] OpenRouter also failed: {e2}")
                self.usage_log.append({"source": "none", "status": "failed"})
                if len(self.usage_log) > 500:
                    self.usage_log.pop(0)
                return {"content": self.FALLBACK_MESSAGE, "source": "none", "status": "failed"}

    def _call_gemini(self, system_prompt: str, user_prompt: str) -> str:
        if not self.gemini_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        url = f"{self.gemini_url}?key={self.gemini_key}"
        payload = {
            "contents": [{"parts": [{"text": f"{system_prompt}\n\nUser: {user_prompt}"}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 4096},
        }
        r = requests.post(url, json=payload, timeout=15)
        r.raise_for_status()
        data = r.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise ValueError("No candidates returned from Gemini.")
        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        if not parts:
            raise ValueError("Content blocked by safety filters or empty response.")
        return parts[0].get("text", "")
