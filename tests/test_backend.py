"""Laya adapter tests without importing Torch or downloading checkpoints."""

from unittest.mock import Mock, patch

import pytest

from app.backend import LayaBackend, ModelSources, NativeHeap
from app.config import Settings


class TestLayaBackend:
    """Verify calls to the library at its public boundary."""

    def test_library_loads_on_first_use_and_is_cached(self) -> None:
        """Ensure constructing the backend does not import Laya until a feature needs it."""
        with patch("app.backend.import_module") as loader:
            backend = LayaBackend()
            loader.assert_not_called()
            assert backend.library is backend.library
            loader.assert_called_once_with("laya")

    @pytest.mark.parametrize(
        "settings, expected_models",
        [
            (Settings(), None),
            (
                Settings(model_id="local"),
                {
                    "english": ("local", None),
                    "multilingual": ("local", "multilingual"),
                    "typed-decisions": ("local", "typed-decisions"),
                },
            ),
            (
                Settings(model_subfolder="custom"),
                {
                    "english": ("convaiinnovations/laya", "custom"),
                    "multilingual": ("convaiinnovations/laya", "multilingual"),
                    "typed-decisions": ("convaiinnovations/laya", "typed-decisions"),
                },
            ),
        ],
    )
    def test_router_factory(self, settings: Settings, expected_models: ModelSources | None) -> None:
        """Verify repository overrides and preload settings reach the library unchanged."""
        with patch("app.backend.import_module") as loader:
            router = LayaBackend().create_router(settings)
            loader.return_value.Router.assert_called_once_with(models=expected_models, device=settings.device)
            loader.return_value.Router.return_value.preload.assert_called_once_with(list(settings.models))
            assert router is loader.return_value.Router.return_value

    def test_preload_skips_weight_init_then_releases_heap(self) -> None:
        """Preload inside `no_init_weights` and trim the heap only after the checkpoints are resident."""
        calls = Mock()
        heap = Mock(spec=NativeHeap)
        heap.release.side_effect = lambda: calls.release()
        with patch("app.backend.import_module") as loader:
            no_init = loader.return_value.no_init_weights.return_value
            no_init.__enter__.side_effect = lambda: calls.enter()
            no_init.__exit__.side_effect = lambda *_: calls.exit()
            loader.return_value.Router.return_value.preload.side_effect = lambda _: calls.preload()
            LayaBackend(heap).create_router(Settings())
        loader.assert_any_call("transformers.initialization")
        assert [name for name, *_ in calls.mock_calls] == ["enter", "preload", "exit", "release"]

    def test_preload_failure_propagates(self) -> None:
        """Ensure checkpoint failures abort startup instead of leaving a partial runtime."""
        heap = Mock(spec=NativeHeap)
        with patch("app.backend.import_module") as loader:
            loader.return_value.Router.return_value.preload.side_effect = RuntimeError("load failed")
            with pytest.raises(RuntimeError, match="load failed"):
                LayaBackend(heap).create_router(Settings())
        heap.release.assert_not_called()

    def test_all_builtin_presets_are_loaded_once(self) -> None:
        """Verify every public preset comes from its Laya factory and is cached."""
        library = Mock()
        with patch("app.backend.import_module", return_value=library):
            backend = LayaBackend()
            presets = backend.presets
            assert backend.presets is presets
        for name in ("triage", "email", "guard", "moderation", "router"):
            factory = getattr(library, f"{name}_questions")
            factory.assert_called_once_with()
            assert presets[name] is factory.return_value


class TestNativeHeap:
    """Verify heap trimming degrades to garbage collection outside glibc."""

    def test_release_trims_when_malloc_trim_exists(self) -> None:
        """Call glibc `malloc_trim(0)` after collecting garbage."""
        with patch("app.backend.gc.collect") as collect, patch("app.backend.ctypes.CDLL") as libc:
            NativeHeap().release()
        collect.assert_called_once_with()
        libc.assert_called_once_with(None)
        libc.return_value.malloc_trim.assert_called_once_with(0)

    def test_release_is_a_no_op_without_malloc_trim(self) -> None:
        """Skip trimming on C libraries such as macOS libSystem that lack `malloc_trim`."""
        with patch("app.backend.ctypes.CDLL", return_value=Mock(spec=[])):
            NativeHeap().release()
