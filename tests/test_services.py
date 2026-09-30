"""Service behavior with a deterministic model boundary."""

import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import Any

import pytest
from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from app.backend import QuestionDefinitions
from app.config import Settings
from app.schemas import (
    BulkItem,
    BulkPredictRequest,
    PredictRequest,
    SystemOneRequest,
    SystemOneResponse,
)
from app.services import SystemOneAdapter
from tests.support import PredictionData, ServiceContext


class TestRuntimeAndMetadata:
    """Router availability follows lifespan and is released on failure."""

    def test_not_started(self) -> None:
        """Report an unloaded runtime and reject model inventory requests before startup."""
        context = ServiceContext()
        health = context.runtime.health()
        assert not health.model_loaded and health.models == []
        with pytest.raises(HTTPException, match="Model not loaded yet") as failure:
            context.runtime.models()
        assert failure.value.status_code == 503

    def test_startup_and_shutdown(self) -> None:
        """Expose configured health metadata while running and release the router on shutdown."""
        context = ServiceContext(Settings(api_keys=("key",), device="mps"))
        with context.running_services():
            assert context.runtime.router is context.router
            assert context.runtime.models().loaded == ["english"]
            health = context.runtime.health()
            assert health.model_loaded and health.auth_enabled and health.models == ["english"]
            assert health.device == "mps" and health.auth_methods == ["apikey"]
        context.backend.create_router.assert_called_once_with(context.settings)
        assert not context.runtime.health().model_loaded

    def test_lifespan_exception_releases_router(self) -> None:
        """Release the model reference even when application execution raises an exception."""
        context = ServiceContext()
        asyncio.run(self._failing_lifespan(context))
        assert not context.runtime.health().model_loaded

    @staticmethod
    async def _failing_lifespan(context: ServiceContext) -> None:
        """Raise inside the lifespan to exercise cleanup after an application failure."""
        with pytest.raises(RuntimeError, match="application failure"):
            async with context.runtime.lifespan(FastAPI()):
                raise RuntimeError("application failure")

    def test_startup_failure_is_not_suppressed(self) -> None:
        """Propagate checkpoint loading errors and keep the runtime unavailable."""
        context = ServiceContext()
        context.backend.create_router.side_effect = RuntimeError("checkpoint unavailable")
        with (
            pytest.raises(RuntimeError, match="checkpoint unavailable"),
            context.running_services(),
        ):
            pytest.fail("Startup should not complete")
        assert not context.runtime.health().model_loaded


class TestQuestions:
    """Question serialization and preset precedence."""

    def test_serialization_excludes_optional_none(self) -> None:
        """Send plain question dictionaries to Laya without unspecified optional fields."""
        context = ServiceContext()
        assert context.predictions.resolve_questions(PredictionData.questions()) == {
            "refund": {"type": "noul", "instructions": "Requesting a refund?"}
        }
        assert context.predictions.resolve_questions(None) == {}

    def test_preset_has_existing_precedence_over_questions(self) -> None:
        """Preserve the established preference for a selected preset over explicit questions."""
        context = ServiceContext()
        assert context.predictions.resolve_questions(PredictionData.questions(), "triage") == {
            "preset": {"type": "noul", "instructions": "triage"}
        }


class TestPrediction:
    """Inference receives resolved data and preserves response extensions."""

    def test_single_prediction(self) -> None:
        """Preserve all validated response fields and forward the requested model override."""
        context = ServiceContext()
        request = PredictRequest(**PredictionData.payload(), model="multilingual")
        with context.running_services():
            result = context.predictions.predict(request)
        assert result.model_dump() == PredictionData.result()
        context.router.predict.assert_called_once_with(
            request.state,
            context.predictions.resolve_questions(request.questions),
            model="multilingual",
        )

    def test_single_prediction_uses_preset(self) -> None:
        """Resolve the selected preset before calling automatic model routing."""
        context = ServiceContext()
        with context.running_services():
            context.predictions.predict(PredictRequest(state="example", preset="triage"))
        context.router.predict.assert_called_once_with("example", context.backend.presets["triage"], model=None)

    def test_single_errors_are_not_converted_to_success(self) -> None:
        """Propagate inference failures and release the shared lock after an exception."""
        context = ServiceContext()
        context.router.predict.side_effect = RuntimeError("inference failed")
        with context.running_services(), pytest.raises(RuntimeError, match="inference failed"):
            context.predictions.predict(PredictRequest(**PredictionData.payload()))
        assert not context.runtime.lock.locked()

    def test_single_response_is_validated(self) -> None:
        """Reject incomplete model responses instead of returning malformed success data."""
        context = ServiceContext()
        context.router.predict.return_value = {"model": "english"}
        with context.running_services(), pytest.raises(ValidationError):
            context.predictions.predict(PredictRequest(**PredictionData.payload()))

    def test_prediction_requires_running_model(self) -> None:
        """Reject inference before startup without invoking the model backend."""
        context = ServiceContext()
        with pytest.raises(HTTPException) as failure:
            context.predictions.predict(PredictRequest(**PredictionData.payload()))
        assert failure.value.status_code == 503
        context.router.predict.assert_not_called()


class TestBulkPrediction:
    """Bulk preserves ordering, fallback rules, error isolation and request limits."""

    def test_per_item_failure_does_not_discard_other_results(self) -> None:
        """Keep successful batch items in order when an intermediate prediction fails."""
        context = ServiceContext()
        context.router.predict.side_effect = [
            PredictionData.result(),
            RuntimeError("bad state"),
            PredictionData.result(),
        ]
        request = BulkPredictRequest(states=["first", "bad", "last"], preset="triage")
        with context.running_services():
            result = context.predictions.predict_bulk(request)
        assert result.count == 3
        assert [item.ok for item in result.results] == [True, False, True]
        assert result.results[1].error == "bad state"
        assert [call.args[0] for call in context.router.predict.call_args_list] == request.states

    def test_invalid_model_response_is_an_item_error(self) -> None:
        """Isolate response-validation failures to their corresponding batch items."""
        context = ServiceContext()
        context.router.predict.side_effect = [{"invalid": True}, PredictionData.result()]
        with context.running_services():
            result = context.predictions.predict_bulk(BulkPredictRequest(states=["bad", "ok"], preset="guard"))
        assert not result.results[0].ok
        assert result.results[1].ok

    def test_item_overrides_and_empty_question_fallback(self) -> None:
        """Respect item overrides while empty question maps fall back to shared questions."""
        context = ServiceContext()
        override = {"custom": {"type": "noul", "instructions": "Custom?"}}
        request = BulkPredictRequest(
            items=[
                BulkItem.model_validate({"state": "override", "questions": override, "model": "multilingual"}),
                BulkItem(state="fallback", questions={}),
            ],
            questions=PredictionData.questions(),
            model="english",
        )
        with context.running_services():
            context.predictions.predict_bulk(request)
        first, second = context.router.predict.call_args_list
        assert first.args == ("override", override) and first.kwargs == {"model": "multilingual"}
        assert second.args == (
            "fallback",
            context.predictions.resolve_questions(PredictionData.questions()),
        )
        assert second.kwargs == {"model": "english"}

    def test_request_preset_retains_precedence_over_item_questions(self) -> None:
        """Preserve preset precedence even when a batch item supplies its own questions."""
        context = ServiceContext()
        request = BulkPredictRequest(
            items=[BulkItem(state="item", questions=PredictionData.questions())], preset="guard"
        )
        with context.running_services():
            context.predictions.predict_bulk(request)
        assert context.router.predict.call_args.args[1] == context.backend.presets["guard"]

    def test_items_can_supply_all_questions(self) -> None:
        """Accept a batch whose items provide questions without a shared question source."""
        context = ServiceContext()
        request = BulkPredictRequest(items=[BulkItem(state="item", questions=PredictionData.questions())])
        with context.running_services():
            assert context.predictions.predict_bulk(request).results[0].ok

    def test_bulk_limit_checked_before_router_availability(self) -> None:
        """Return the size-limit error before attempting to use an unloaded model."""
        context = ServiceContext(Settings(max_bulk_items=1))
        request = BulkPredictRequest(states=["one", "two"], preset="triage")
        with pytest.raises(HTTPException) as failure:
            context.predictions.predict_bulk(request)
        assert failure.value.status_code == 422
        assert failure.value.detail == "Too many states: 2 > MAX_BULK_ITEMS=1"
        context.router.predict.assert_not_called()

    def test_exact_limit_is_allowed(self) -> None:
        """Accept a batch whose item count equals the configured maximum."""
        context = ServiceContext(Settings(max_bulk_items=2))
        with context.running_services():
            result = context.predictions.predict_bulk(BulkPredictRequest(states=["one", "two"], preset="triage"))
        assert result.count == 2


class TestSystemOne:
    """Model aliases and response adaptation remain compatible with TypeSafe clients."""

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("english", "english"),
            ("laya-english", "english"),
            ("typesafe/laya-multilingual", "multilingual"),
            ("laya/typed-decisions", "typed-decisions"),
            (" TYPESAFE/LAYA/LAYA-TYPED-DECISIONS ", "typed-decisions"),
            ("custom-multi-v2", "multilingual"),
            ("custom-decision-v2", "typed-decisions"),
            ("unknown", "english"),
        ],
    )
    def test_alias_resolution(self, name: str, expected: str) -> None:
        """Normalize supported aliases and retain fallback routing for unfamiliar names."""
        assert SystemOneAdapter.resolve_model(name) == expected

    def test_response_keeps_external_model_name_and_omits_laya_only_fields(self) -> None:
        """Preserve the client model identifier while removing native-only answer metadata."""
        context = ServiceContext()
        request = SystemOneRequest(**PredictionData.payload(), model="typesafe/laya-multilingual")
        with context.running_services():
            result = context.systemone.predict(request)
        assert result.model == request.model and result.provider == "Laya"
        assert re.fullmatch(r"gen-laya-[0-9a-f]{24}", result.id)
        assert all("action" not in answer.model_dump() for answer in result.answers.values())
        assert result.answers["refund"].model_dump() == {"type": "noul", "noul": 0.9}
        assert context.router.predict.call_args.kwargs == {"model": "multilingual"}

    def test_identifiers_are_unique(self) -> None:
        """Generate a distinct identifier for each SystemOne prediction response."""
        payload = {
            "model": "english",
            "answers": {},
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
        assert SystemOneResponse.model_validate(payload).id != SystemOneResponse.model_validate(payload).id


class InferenceProbe:
    """Block the first inference so competing calls can be observed deterministically."""

    def __init__(self) -> None:
        """Prepare synchronization events and an ordered record of inference calls."""
        self.entered = Event()
        self.release = Event()
        self.states: list[str] = []

    def predict(self, state: str, questions: QuestionDefinitions, *, model: str | None = None) -> dict[str, Any]:
        """Pause the first inference until the test allows the batch to continue."""
        self.states.append(state)
        if state == "first":
            self.entered.set()
            assert self.release.wait(3), "Test did not release the first inference"
        return PredictionData.result()


class TestInferenceSerialization:
    """Single and SystemOne requests cannot interleave inside a bulk request."""

    def test_bulk_holds_shared_lock_for_entire_batch(self) -> None:
        """Prevent native and SystemOne requests from interleaving inside an active batch."""
        context, probe = ServiceContext(), InferenceProbe()
        context.router.predict.side_effect = probe.predict
        with context.running_services(), ThreadPoolExecutor(max_workers=3) as executor:
            batch = executor.submit(
                context.predictions.predict_bulk,
                BulkPredictRequest(states=["first", "second"], preset="triage"),
            )
            try:
                assert probe.entered.wait(3)
                assert context.runtime.lock.locked()
                single = executor.submit(context.predictions.predict, PredictRequest(state="single", preset="triage"))
                compatible = executor.submit(
                    context.systemone.predict,
                    SystemOneRequest(state="systemone", model="english", questions={}),
                )
            finally:
                probe.release.set()
            assert batch.result(timeout=3).count == 2
            single.result(timeout=3)
            compatible.result(timeout=3)
        assert probe.states[:2] == ["first", "second"]
        assert set(probe.states[2:]) == {"single", "systemone"}
