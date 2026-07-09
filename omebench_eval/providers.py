"""Model providers for API-based evaluation (Anthropic + OpenAI).

Each provider exposes `.generate(system, user) -> str`. Friendly model names
(e.g. "opus", "gpt-5.5") are resolved via MODEL_REGISTRY, but any concrete
model id can be passed through with `--model-id`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class ModelSpec:
    provider: str  # "anthropic" | "openai"
    model_id: str
    # Optional reasoning controls (used only if the provider/model supports them).
    reasoning_effort: Optional[str] = None  # openai: "low"|"medium"|"high"
    thinking_budget: Optional[int] = None  # anthropic extended-thinking token budget
    extra: Dict = field(default_factory=dict)


# Friendly aliases -> default concrete model ids. Override any id with --model-id.
# Model ids are best-effort defaults; adjust to whatever your account exposes.
MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "opus": ModelSpec("anthropic", "claude-opus-4-6", thinking_budget=None),
    "opus-thinking": ModelSpec("anthropic", "claude-opus-4-6", thinking_budget=8000),
    "sonnet": ModelSpec("anthropic", "claude-sonnet-4-6"),
    "gpt-5.5": ModelSpec("openai", "gpt-5.5", reasoning_effort="medium"),
    "gpt-5.5-high": ModelSpec("openai", "gpt-5.5", reasoning_effort="high"),
    "gpt-5.2": ModelSpec("openai", "gpt-5.2", reasoning_effort="medium"),
}


def resolve_spec(
    name: str,
    model_id: Optional[str] = None,
    provider: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    thinking_budget: Optional[int] = None,
) -> ModelSpec:
    """Resolve a friendly name (or raw provider/model-id) into a ModelSpec."""
    if name in MODEL_REGISTRY:
        spec = MODEL_REGISTRY[name]
        spec = ModelSpec(
            provider=provider or spec.provider,
            model_id=model_id or spec.model_id,
            reasoning_effort=reasoning_effort or spec.reasoning_effort,
            thinking_budget=thinking_budget if thinking_budget is not None else spec.thinking_budget,
        )
        return spec
    if not (provider and model_id):
        raise ValueError(
            f"Unknown model alias '{name}'. Either use a known alias "
            f"({list(MODEL_REGISTRY)}) or pass both --provider and --model-id."
        )
    return ModelSpec(
        provider=provider,
        model_id=model_id,
        reasoning_effort=reasoning_effort,
        thinking_budget=thinking_budget,
    )


class BaseProvider:
    def __init__(self, spec: ModelSpec, max_tokens: int = 4096, temperature: float = 0.0):
        self.spec = spec
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, system: str, user: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class AnthropicProvider(BaseProvider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            import anthropic
        except ImportError as e:
            raise ImportError("Install the Anthropic SDK: pip install anthropic") from e
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        self._client = anthropic.Anthropic()

    def generate(self, system: str, user: str) -> str:
        kwargs = dict(
            model=self.spec.model_id,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if self.spec.thinking_budget:
            # Extended thinking requires temperature=1 and budget < max_tokens.
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.spec.thinking_budget,
            }
            kwargs["max_tokens"] = max(self.max_tokens, self.spec.thinking_budget + 1024)
        else:
            kwargs["temperature"] = self.temperature

        resp = self._client.messages.create(**kwargs)
        # Concatenate all text blocks (skip thinking blocks).
        parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        return "".join(parts).strip()


class OpenAIProvider(BaseProvider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            import openai  # noqa: F401
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("Install the OpenAI SDK: pip install openai") from e
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self._client = OpenAI()

    def generate(self, system: str, user: str) -> str:
        # Prefer the Responses API (works with reasoning models); fall back to
        # chat.completions if unavailable.
        try:
            return self._via_responses(system, user)
        except Exception:
            return self._via_chat(system, user)

    def _via_responses(self, system: str, user: str) -> str:
        kwargs = dict(
            model=self.spec.model_id,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_output_tokens=self.max_tokens,
        )
        if self.spec.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.spec.reasoning_effort}
        else:
            kwargs["temperature"] = self.temperature
        resp = self._client.responses.create(**kwargs)
        text = getattr(resp, "output_text", None)
        if text:
            return text.strip()
        # Manual extraction fallback.
        chunks = []
        for item in getattr(resp, "output", []) or []:
            for c in getattr(item, "content", []) or []:
                if getattr(c, "type", None) in ("output_text", "text"):
                    chunks.append(getattr(c, "text", ""))
        return "".join(chunks).strip()

    def _via_chat(self, system: str, user: str) -> str:
        kwargs = dict(
            model=self.spec.model_id,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=self.max_tokens,
        )
        # Reasoning models often reject a custom temperature; only send it when
        # no reasoning effort is configured.
        if not self.spec.reasoning_effort:
            kwargs["temperature"] = self.temperature
        resp = self._client.chat.completions.create(**kwargs)
        return (resp.choices[0].message.content or "").strip()


class LocalHFProvider(BaseProvider):
    """Run a local/HF checkpoint (e.g. your fine-tuned specialist) via transformers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Install the training/eval stack: pip install -r requirements-train.txt"
            ) from e
        self._tok = AutoTokenizer.from_pretrained(self.spec.model_id, trust_remote_code=True)
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            self.spec.model_id,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )

    def generate(self, system: str, user: str) -> str:
        import torch

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        text = self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tok([text], return_tensors="pt").to(self._model.device)
        do_sample = self.temperature and self.temperature > 0
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=bool(do_sample),
                temperature=self.temperature if do_sample else None,
                pad_token_id=self._tok.pad_token_id or self._tok.eos_token_id,
            )
        gen = out[0][inputs.input_ids.shape[1]:]
        return self._tok.decode(gen, skip_special_tokens=True).strip()


def make_provider(spec: ModelSpec, max_tokens: int, temperature: float) -> BaseProvider:
    if spec.provider == "anthropic":
        return AnthropicProvider(spec, max_tokens=max_tokens, temperature=temperature)
    if spec.provider == "openai":
        return OpenAIProvider(spec, max_tokens=max_tokens, temperature=temperature)
    if spec.provider == "local":
        return LocalHFProvider(spec, max_tokens=max_tokens, temperature=temperature)
    raise ValueError(f"Unknown provider '{spec.provider}'.")
