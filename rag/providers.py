"""One interface for every model backend: generate(prompt) -> str.

Local (LiteRT/Gemma) is the default. Cloud is only reachable with explicit consent,
and there is never a silent fallback from local to cloud.
"""
import logging
import os
import threading
from typing import Callable, Dict, List, Optional, Tuple

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
_engine_cache: Dict[Tuple[str, str], Tuple[object, str]] = {}
_engine_lock = threading.Lock()
_gpu_disabled = False  # set after the GPU is lost once; the rest of the process then uses the CPU
_GPU_FAILURE_HINTS = ("device is lost", "device lost", "device removed", "internal")


def _load_engine(model_file: str, backend: str = "auto") -> Tuple[object, str]:
    """Return (engine, backend actually in use), loading the model once per file and backend.

    backend: "auto" tries the GPU and falls back to the CPU, "gpu" insists on the GPU, "cpu" skips it.
    A UI rerun must not reload the model, so engines are cached.
    """
    if _gpu_disabled and backend == "auto":
        backend = "cpu"
    key = (model_file, "cpu" if backend == "cpu" else "gpu")
    with _engine_lock:
        if key in _engine_cache:
            return _engine_cache[key]
        try:
            import litert_lm
        except ImportError as e:
            raise ProviderError("litert-lm-api is not installed. Run: pip install -r requirements.txt") from e
        engine, used = None, "cpu"
        if backend != "cpu":
            try:
                engine, used = litert_lm.Engine(model_file, backend=litert_lm.Backend.GPU()), "gpu"
            except Exception as gpu_err:
                if backend == "gpu":
                    raise ProviderError(
                        f"Could not start the local model on the GPU: {gpu_err}. "
                        "Set RAG_LITERT_BACKEND=cpu to run it on the CPU."
                    ) from gpu_err
                log.warning("GPU backend unavailable (%s); using CPU for the local model.", gpu_err)
        if engine is None:
            try:
                engine = litert_lm.Engine(model_file, backend=litert_lm.Backend.CPU())
            except Exception as cpu_err:
                raise ProviderError(
                    f"Could not load the local model {model_file}: {cpu_err}. "
                    "Check that the file is complete and that you have enough free RAM."
                ) from cpu_err
        _engine_cache[key] = (engine, used)
        return engine, used


def _evict_gpu_engine(model_file: str) -> None:
    """Forget the GPU engine of a model. A CPU engine that another session already loaded stays cached."""
    with _engine_lock:
        _engine_cache.pop((model_file, "gpu"), None)


def _looks_like_gpu_failure(error: Exception) -> bool:
    return any(hint in str(error).lower() for hint in _GPU_FAILURE_HINTS)


class LocalProvider:
    def __init__(self):
        self.model_file = os.path.join(CONFIG.model_dir, f"{CONFIG.litert_model}.litertlm")
        if not os.path.isfile(self.model_file):
            raise ProviderError(
                f"Model file not found: {self.model_file}. Download it from Hugging Face "
                f"(litert-community/{CONFIG.litert_model}-litert-lm) into {CONFIG.model_dir}, "
                "or set RAG_MODEL_DIR / RAG_LITERT_MODEL."
            )
        self._engine, self.backend = _load_engine(self.model_file, CONFIG.litert_backend)
        self._set_name()

    def _set_name(self) -> None:
        self.name = f"litert:{CONFIG.litert_model}" + (" (cpu)" if self.backend == "cpu" else "")

    def _generate(self, prompt: str) -> str:
        with self._engine.create_conversation(max_output_tokens=CONFIG.max_output_tokens) as conv:
            reply = conv.send_message(prompt)
            return reply["content"][0]["text"].strip()

    def __call__(self, prompt: str) -> str:
        global _gpu_disabled
        try:
            return self._generate(prompt)
        except ProviderError:
            raise
        except Exception as e:
            if self.backend == "cpu" or not _looks_like_gpu_failure(e):
                raise ProviderError(f"The local model failed while answering: {e}") from e
            # The GPU was lost (it is shared with other programs and can be reset by the driver). A lost
            # engine never recovers, so drop it, reload on the CPU and retry this one question.
            log.warning("The GPU failed (%s); reloading the local model on the CPU and retrying once.", e)
            _gpu_disabled = True
            _evict_gpu_engine(self.model_file)
            self._engine, self.backend = _load_engine(self.model_file, "cpu")
            self._set_name()
            try:
                return self._generate(prompt)
            except Exception as retry_error:
                raise ProviderError(
                    f"The local model failed while answering, also on the CPU: {retry_error}"
                ) from retry_error


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
