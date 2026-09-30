"""Lazy boundary to Laya, so the OpenAPI schema builds without torch installed."""

import ctypes
import gc
from collections.abc import Iterable
from functools import cached_property
from importlib import import_module
from types import ModuleType
from typing import Any, Protocol, get_args

from app.config import AVAILABLE_MODELS, DEFAULT_MODEL_ID, Settings
from app.schemas import ModelName, PresetName, State

QuestionDefinitions = dict[str, dict[str, Any]]
ModelSources = dict[str, tuple[str, str | None]]


class PredictionRouter(Protocol):
    """The part of `laya.Router` the API relies on."""

    @property
    def loaded(self) -> Iterable[str]:
        """Checkpoints currently resident in memory."""
        ...

    def preload(self, models: list[str]) -> object:
        """Load checkpoints before serving requests."""
        ...

    def predict(
        self, state: State, questions: QuestionDefinitions, *, model: ModelName | None = None
    ) -> dict[str, Any]:
        """Answer the questions for one state, auto-routed unless `model` is pinned."""
        ...


class NativeHeap:
    """Returns memory freed after checkpoint loading to the operating system."""

    def release(self) -> None:
        """Collect garbage, then trim the glibc heap; a no-op where `malloc_trim` does not exist."""
        gc.collect()
        # glibc keeps the freed load buffers mapped, holding ~300 MB of RSS a
        # 2 GB container needs for inference.
        trim = getattr(ctypes.CDLL(None), "malloc_trim", None)
        if trim is not None:
            trim(0)


class LayaBackend:
    """The only place that touches the laya module."""

    def __init__(self, heap: NativeHeap | None = None) -> None:
        """Use the given heap to release load buffers, or the process heap by default."""
        self._heap = heap or NativeHeap()

    @cached_property
    def library(self) -> ModuleType:
        """The laya module, imported on first use."""
        # CI generates openapi.json with only fastapi and pydantic installed.
        return import_module("laya")

    @cached_property
    def initialization(self) -> ModuleType:
        """The transformers weight-initialization module, imported on first use."""
        return import_module("transformers.initialization")

    @cached_property
    def presets(self) -> dict[str, QuestionDefinitions]:
        """Built-in question sets, keyed by preset name."""
        return {name: getattr(self.library, f"{name}_questions")() for name in get_args(PresetName)}

    def create_router(self, settings: Settings) -> PredictionRouter:
        """Build the router and preload the configured checkpoints."""
        router: PredictionRouter = self.library.Router(models=self._model_sources(settings), device=settings.device)
        # Laya builds each encoder with `from_config` and then overwrites every weight from the
        # checkpoint. Transformers would first randomly initialise all of them, which peaks above
        # 3.7 GB for the multilingual encoder (256k-token vocabulary) and gets the container OOM-killed.
        with self.initialization.no_init_weights():
            router.preload(list(settings.models))
        self._heap.release()
        return router

    @staticmethod
    def _model_sources(settings: Settings) -> ModelSources | None:
        # None keeps Laya's bundled defaults; the subfolder override applies to English only.
        if settings.model_id == DEFAULT_MODEL_ID and not settings.model_subfolder:
            return None
        return {
            name: (settings.model_id, settings.model_subfolder if name == "english" else name)
            for name in AVAILABLE_MODELS
        }
