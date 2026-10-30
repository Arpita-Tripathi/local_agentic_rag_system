"""One interface for every model backend: generate(prompt) -> str.

Local (LiteRT/Gemma) is the default. Cloud is only reachable with explicit consent,
and there is never a silent fallback from local to cloud.
"""
import logging
import os
import threading
from typing import Callable, Dict, List, Optional

from rag.config import CONFIG

log = logging.getLogger(__name__)

Generate = Callable[[str], str]

_CLOUD_URL = "https://openrouter.ai/api/v1/chat/completions"


class ProviderError(Exception):
    """User-readable failure from a model backend."""


class ConsentRequired(ProviderError):
    """Cloud mode was requested without the user agreeing to send excerpts."""


class FakeProvider:
    """Deterministic provider for tests: returns queued replies in order."""

    name = "fake"

    def __init__(self, replies: List[str]):
        self.replies = list(replies)
        self.prompts: List[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.replies:
            raise ProviderError("FakeProvider has no replies left.")
        return self.replies.pop(0)


# ---------- local ----------
_engine_cache: Dict[str, object] = {}
_engine_lock = threading.Lock()


def _load_engine(model_file: str):
    """Load the LiteRT engine once per model file (a UI rerun must not reload the model)."""
    with _engine_lock:
        if model_file in _engine_cache:
            return _engine_cache[model_file]
        try:
            import litert_lm
        except ImportError as e:
            raise ProviderError("litert-lm-api is not installed. Run: pip install -r requirements.txt") from e
        try:
            engine = litert_lm.Engine(model_file, backend=litert_lm.Backend.GPU())
        except Exception as gpu_err:
            log.warning("GPU backend unavailable (%s); using CPU for the local model.", gpu_err)
            try:
                engine = litert_lm.Engine(model_file, backend=litert_lm.Backend.CPU())
            except Exception as cpu_err:
                raise ProviderError(
                    f"Could not load the local model {model_file}: {cpu_err}. "
                    "Check that the file is complete and that you have enough free RAM."
                ) from cpu_err
        _engine_cache[model_file] = engine
        return engine


class LocalProvider:
    def __init__(self):
        self.model_file = os.path.join(CONFIG.model_dir, f"{CONFIG.litert_model}.litertlm")
        self.name = f"litert:{CONFIG.litert_model}"
        if not os.path.isfile(self.model_file):
            raise ProviderError(
                f"Model file not found: {self.model_file}. Download it from Hugging Face "
                f"(litert-community/{CONFIG.litert_model}-litert-lm) into {CONFIG.model_dir}, "
                "or set RAG_MODEL_DIR / RAG_LITERT_MODEL."
            )
        self._engine = _load_engine(self.model_file)

    def __call__(self, prompt: str) -> str:
        try:
            with self._engine.create_conversation(max_output_tokens=CONFIG.max_output_tokens) as conv:
                reply = conv.send_message(prompt)
                return reply["content"][0]["text"].strip()
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(f"The local model failed while answering: {e}") from e


# ---------- cloud ----------
class CloudProvider:
    """Sends the prompt (retrieved excerpts only) to the cloud. Tracks usage for the UI."""

    def __init__(self):
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key or not CONFIG.cloud_model:
            raise ProviderError(
                "Cloud mode needs OPENROUTER_API_KEY and RAG_CLOUD_MODEL in your environment."
            )
        try:
            import requests  # noqa: F401
        except ImportError as e:
            raise ProviderError("requests is not installed. Run: pip install -r requirements.txt") from e
        self._key = key
        self.name = f"cloud:{CONFIG.cloud_model}"
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def __call__(self, prompt: str) -> str:
        import requests

        log.info("Sending excerpts to the cloud provider (%s).", CONFIG.cloud_model)
        try:
            res = requests.post(
                _CLOUD_URL,
                headers={"Authorization": f"Bearer {self._key}"},
                json={
                    "model": CONFIG.cloud_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": CONFIG.max_output_tokens,
                },
                timeout=CONFIG.cloud_timeout_s,
            )
        except requests.RequestException as e:
            raise ProviderError(f"Network error talking to the cloud provider: {e}") from e

        if res.status_code == 429:
            raise ProviderError("Cloud provider rate limit reached (429). Wait a bit and try again.")
        if res.status_code in (401, 403):
            raise ProviderError("The cloud provider rejected the API key. Check OPENROUTER_API_KEY.")
        if res.status_code != 200:
            raise ProviderError(f"Cloud provider error {res.status_code}: {res.text[:200]}")

        try:
            data = res.json()
            text = data["choices"][0]["message"]["content"].strip()
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as e:
            raise ProviderError("The cloud provider returned an unexpected response.") from e

        usage = data.get("usage") or {}
        self.usage["calls"] += 1
        self.usage["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        self.usage["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        return text


def get_provider(mode: Optional[str] = None, cloud_consent: bool = False) -> Generate:
    mode = mode or CONFIG.mode
    if mode == "local":
        return LocalProvider()
    if mode == "cloud":
        if not cloud_consent:
            raise ConsentRequired("Cloud mode sends PDF excerpts off this device. Ask the user first.")
        return CloudProvider()
    raise ProviderError(f"Unknown mode: {mode!r}")
