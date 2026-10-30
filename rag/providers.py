"""One interface for every model backend: generate(prompt) -> str.

Local (LiteRT/Gemma) is the default. Cloud is only reachable with explicit consent,
and there is never a silent fallback from local to cloud.
"""
import os
from typing import Callable, List, Optional

from rag.config import CONFIG

Generate = Callable[[str], str]


class ProviderError(Exception):
    """User-readable failure from a model backend."""


class ConsentRequired(ProviderError):
    """Cloud mode was requested without the user agreeing to send excerpts."""


class FakeProvider:
    """Deterministic provider for tests: returns queued replies in order."""

    def __init__(self, replies: List[str]):
        self.replies = list(replies)
        self.prompts: List[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.replies:
            raise ProviderError("FakeProvider has no replies left.")
        return self.replies.pop(0)


def _local_provider() -> Generate:
    try:
        import litert_lm
    except ImportError as e:
        raise ProviderError("litert-lm-api is not installed. Run: pip install litert-lm-api") from e

    model_file = os.path.join(CONFIG.model_dir, f"{CONFIG.litert_model}.litertlm")
    if not os.path.isfile(model_file):
        raise ProviderError(
            f"Model file not found: {model_file}. Download it from Hugging Face "
            f"(litert-community/{CONFIG.litert_model}-litert-lm) into {CONFIG.model_dir}."
        )

    try:
        engine = litert_lm.Engine(model_file, backend=litert_lm.Backend.GPU())
    except Exception:
        engine = litert_lm.Engine(model_file, backend=litert_lm.Backend.CPU())

    def generate(prompt: str) -> str:
        with engine.create_conversation(max_output_tokens=CONFIG.max_output_tokens) as conv:
            reply = conv.send_message(prompt)
            return reply["content"][0]["text"].strip()

    return generate


def _cloud_provider() -> Generate:
    import requests

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key or not CONFIG.cloud_model:
        raise ProviderError("Cloud mode needs OPENROUTER_API_KEY and RAG_CLOUD_MODEL in your environment.")

    def generate(prompt: str) -> str:
        try:
            res = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": CONFIG.cloud_model, "messages": [{"role": "user", "content": prompt}]},
                timeout=60,
            )
        except requests.RequestException as e:
            raise ProviderError(f"Network error talking to the cloud provider: {e}") from e
        if res.status_code == 429:
            raise ProviderError("Cloud provider rate limit reached (429). Wait a bit and try again.")
        if res.status_code != 200:
            raise ProviderError(f"Cloud provider error {res.status_code}: {res.text[:200]}")
        return res.json()["choices"][0]["message"]["content"].strip()

    return generate


def get_provider(mode: Optional[str] = None, cloud_consent: bool = False) -> Generate:
    mode = mode or CONFIG.mode
    if mode == "local":
        return _local_provider()
    if mode == "cloud":
        if not cloud_consent:
            raise ConsentRequired("Cloud mode sends PDF excerpts off this device. Ask the user first.")
        return _cloud_provider()
    raise ProviderError(f"Unknown mode: {mode!r}")
