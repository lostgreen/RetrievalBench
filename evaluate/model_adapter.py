"""Model adapter that wraps APIBank and OpenAI-compatible clients."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping
import re

from evaluate.config import EvalConfig
from evaluate.evaluator import ModelInterface
from evaluate.utils import LOGGER, ensure_api_bank


class APIBankModelAdapter(ModelInterface):
    """Adapter that uses APIBank to rotate Gemini API keys.

    Exposes a simple `predict` method compatible with our evaluation pipeline.
    """

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

    def __init__(self, cfg: EvalConfig):
        super().__init__(name=cfg.model_name)
        self.cfg = cfg

        try:
            from APIBank import build_default_bank  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - configuration issue
            raise ImportError(
                "APIBank package is not available on PYTHONPATH; "
                "add the APIBank directory or install it as a package."
            ) from exc

        self._bank = build_default_bank()

        try:
            from openai import OpenAI  # type: ignore
        except ImportError:  # pragma: no cover - optional dependency
            LOGGER.warning(
                "The 'openai' package is not installed. Falling back to heuristic predictions."
            )
            self._client_cls = None
        else:
            self._client_cls = OpenAI

    @staticmethod
    def _fallback_order(shot_count: int) -> list[int]:
        return list(range(shot_count))

    @staticmethod
    def _extract_indices(text: str, shot_count: int) -> list[int]:
        """Extract shot indices from model output.

        Strategy (in order):
        1) Match the last occurrence of: `Shots: [i, j, ...]` (case-insensitive, multi-line).
        2) Match the last occurrence of: `Shots: 1, 2, 3` (no brackets; read to end-of-line).
        3) Fallback: conservative standalone integers from the whole text.
        All results are de-duplicated in order and clipped to [0, shot_count).
        """
        # 1) Bracketed form
        pat_bracket = re.compile(r"shots\s*[:\-]?\s*\[([^\]]*?)\]", re.IGNORECASE | re.DOTALL)
        matches = list(pat_bracket.finditer(text))
        if matches:
            inner = matches[-1].group(1)
            nums = re.findall(r"-?\d+", inner)
            out: list[int] = []
            for n in nums:
                v = int(n)
                if 0 <= v < shot_count and v not in out:
                    out.append(v)
            if out:
                return out

        # 2) Non-bracket form on the line after 'Shots:' (to EOL)
        pat_line = re.compile(r"shots\s*[:\-]\s*([^\n\r]*)", re.IGNORECASE)
        matches = list(pat_line.finditer(text))
        if matches:
            tail = matches[-1].group(1)
            nums = re.findall(r"-?\d+", tail)
            out: list[int] = []
            for n in nums:
                v = int(n)
                if 0 <= v < shot_count and v not in out:
                    out.append(v)
            if out:
                return out

        # 3) Conservative fallback
        out: list[int] = []
        for n in re.findall(r"\b\d+\b", text):
            v = int(n)
            if 0 <= v < shot_count and v not in out:
                out.append(v)
        if not out:
            out = list(range(shot_count))
        return out

    def predict(self, prompt: str, **kwargs: Any) -> Mapping[str, Any]:  # type: ignore[override]
        shot_count = int(kwargs.get("shot_count", 0)) or 0
        video_path = kwargs.get("video_path", "unknown")
        messages = kwargs.get("messages", None)

        if self._client_cls is None:
            return {
                "video_path": video_path,
                "shot_order": self._fallback_order(shot_count),
                "explanation": "openai package missing; using fallback ordering",
            }

        if messages is None:
            # Original behavior if messages are not provided
            question = kwargs.get("question")
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that selects relevant video shots.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
            if question:
                messages.append({"role": "user", "content": f"Question: {question}"})

        with self._bank.checkout() as key:
            client = self._client_cls(api_key=key.secret, base_url=self.BASE_URL)

            try:
                response = client.chat.completions.create(  # type: ignore[attr-defined]
                    model=self.cfg.model_name,
                    messages=messages,
                    # max_tokens=1024,
                )
            except Exception as error:  # pragma: no cover - depends on external API
                LOGGER.error("Model request failed: %s", error)
                self._bank.report_failure(key)
                return {
                    "video_path": video_path,
                    "shot_order": self._fallback_order(shot_count),
                    "error": str(error),
                }

            choice = response.choices[0].message  # type: ignore[index]
            content = getattr(choice, "content", "") or ""

            if kwargs.get("is_round_1", False):
                indices = self._extract_indices(content, shot_count)
                return {
                    "video_path": video_path,
                    "shot_order": indices,
                    "raw_completion": content,
                    "used_key": key.alias or key.fingerprint,
                }
            else:
                return {
                    "video_path": video_path,
                    "final_response": content,
                    "raw_completion": content,
                    "used_key": key.alias or key.fingerprint,
                }


def load_model(cfg: EvalConfig) -> ModelInterface:
    """Construct the model adapter after ensuring APIBank is available."""
    api_root = ensure_api_bank(cfg)
    parent_dir = api_root.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    return APIBankModelAdapter(cfg)
