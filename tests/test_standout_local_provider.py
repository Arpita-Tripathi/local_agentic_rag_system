"""The local model survives a lost GPU: reload on the CPU, retry once, and remember."""
import sys
import types

import pytest

from rag import providers
from rag.config import Config
from rag.providers import LocalProvider, ProviderError


class FakeLiteRT(types.ModuleType):
    """Stands in for litert_lm: the GPU engine can be made to lose its device on the first message."""

    def __init__(self, gpu_works=True, gpu_fails_to_start=False, cpu_error=None):
        super().__init__("litert_lm")
        self.created, self.messages = [], []
        outer = self

        class Backend:
            @staticmethod
            def GPU():
                return "gpu"

            @staticmethod
            def CPU():
                return "cpu"

        class Conversation:
            def __init__(self, engine):
                self.engine = engine

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def send_message(self, prompt):
                outer.messages.append(self.engine.backend)
                if self.engine.backend == "gpu" and not gpu_works:
                    raise RuntimeError("INTERNAL: INTERNAL: [Device] is lost.")
                if self.engine.backend == "cpu" and cpu_error:
                    raise RuntimeError(cpu_error)
                return {"content": [{"text": f" answer from {self.engine.backend} "}]}

        class Engine:
            def __init__(self, path, backend):
                if backend == "gpu" and gpu_fails_to_start:
                    raise RuntimeError("no GPU here")
                self.backend = backend
                outer.created.append(backend)

            def create_conversation(self, max_output_tokens=None):
                return Conversation(self)

        self.Backend, self.Engine = Backend, Engine


@pytest.fixture
def setup(monkeypatch, tmp_path):
    (tmp_path / "m.litertlm").write_bytes(b"x")

    def install(fake, backend="auto"):
        monkeypatch.setitem(sys.modules, "litert_lm", fake)
        monkeypatch.setattr(providers, "CONFIG", Config(model_dir=str(tmp_path), litert_model="m",
                                                        litert_backend=backend))
        monkeypatch.setattr(providers, "_engine_cache", {})
        monkeypatch.setattr(providers, "_gpu_disabled", False)
        return fake

    return install


def test_gpu_is_used_when_it_works(setup):
    fake = setup(FakeLiteRT())
    provider = LocalProvider()
    assert provider("q") == "answer from gpu" and provider.name == "litert:m" and fake.created == ["gpu"]


def test_lost_gpu_reloads_on_the_cpu_and_retries_once(setup):
    fake = setup(FakeLiteRT(gpu_works=False))
    provider = LocalProvider()
    assert provider("q") == "answer from cpu"
    assert fake.messages == ["gpu", "cpu"] and fake.created == ["gpu", "cpu"]
    assert provider.backend == "cpu" and provider.name == "litert:m (cpu)"


def test_after_a_lost_gpu_later_questions_stay_on_the_cpu_without_reloading(setup):
    fake = setup(FakeLiteRT(gpu_works=False))
    provider = LocalProvider()
    provider("first")
    provider("second")
    assert fake.created == ["gpu", "cpu"] and fake.messages == ["gpu", "cpu", "cpu"]


def test_a_new_provider_after_a_lost_gpu_goes_straight_to_the_cpu(setup):
    fake = setup(FakeLiteRT(gpu_works=False))
    LocalProvider()("first")
    fresh = LocalProvider()  # e.g. another browser session
    assert fresh.backend == "cpu" and fresh("again") == "answer from cpu"
    assert fake.created == ["gpu", "cpu"]  # the CPU engine was reused, the GPU not tried again


def test_other_sessions_holding_the_dead_engine_recover_too(setup):
    fake = setup(FakeLiteRT(gpu_works=False))
    first, second = LocalProvider(), LocalProvider()
    first("q")
    assert second("q") == "answer from cpu" and fake.created == ["gpu", "cpu"]


def test_a_failure_that_is_not_about_the_gpu_is_reported_not_retried(setup):
    class Boom(FakeLiteRT):
        pass

    fake = setup(FakeLiteRT(cpu_error="tokenizer exploded"), backend="cpu")
    with pytest.raises(ProviderError, match="tokenizer exploded"):
        LocalProvider()("q")
    assert fake.messages == ["cpu"]


def test_failing_again_on_the_cpu_is_one_clear_error(setup):
    fake = setup(FakeLiteRT(gpu_works=False, cpu_error="out of memory"))
    with pytest.raises(ProviderError, match="also on the CPU: out of memory"):
        LocalProvider()("q")
    assert fake.messages == ["gpu", "cpu"]


def test_backend_cpu_never_touches_the_gpu(setup):
    fake = setup(FakeLiteRT(gpu_works=False), backend="cpu")
    provider = LocalProvider()
    assert provider("q") == "answer from cpu" and fake.created == ["cpu"] and provider.name.endswith("(cpu)")


def test_auto_falls_back_to_the_cpu_when_the_gpu_cannot_start(setup):
    fake = setup(FakeLiteRT(gpu_fails_to_start=True))
    assert LocalProvider()("q") == "answer from cpu" and fake.created == ["cpu"]


def test_backend_gpu_that_cannot_start_says_how_to_use_the_cpu(setup):
    setup(FakeLiteRT(gpu_fails_to_start=True), backend="gpu")
    with pytest.raises(ProviderError, match="RAG_LITERT_BACKEND=cpu"):
        LocalProvider()


def test_backend_setting_is_validated(monkeypatch):
    monkeypatch.setenv("RAG_LITERT_BACKEND", "tpu")
    with pytest.raises(ValueError, match="RAG_LITERT_BACKEND"):
        Config()
    monkeypatch.setenv("RAG_LITERT_BACKEND", " CPU ")
    assert Config().litert_backend == "cpu"
    monkeypatch.delenv("RAG_LITERT_BACKEND")
    assert Config().litert_backend == "auto"
