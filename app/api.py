"""HTTP endpoints. Handler docstrings are the operation descriptions in the OpenAPI schema."""

from typing import Any, cast

from fastapi import APIRouter, Depends, FastAPI

from app.authentication import Authenticator
from app.backend import LayaBackend
from app.schemas import (
    BulkPredictRequest,
    BulkPredictResponse,
    DetectRequest,
    EmailStateRequest,
    HealthResponse,
    ModelsResponse,
    PredictRequest,
    PredictResponse,
    SystemOneRequest,
    SystemOneResponse,
)
from app.services import ModelRuntime, PredictionService, SystemOneAdapter

AUTH_RESPONSES: dict[int | str, dict[str, Any]] = {
    401: {"description": "Missing or invalid credentials"},
    503: {"description": "Model not loaded yet"},
}


class ApiRoutes:
    """Binds the HTTP contract to the application's services."""

    def __init__(
        self,
        runtime: ModelRuntime,
        predictions: PredictionService,
        systemone: SystemOneAdapter,
        backend: LayaBackend,
    ) -> None:
        self._runtime = runtime
        self._predictions = predictions
        self._systemone = systemone
        self._backend = backend

    def register(self, app: FastAPI, authenticator: Authenticator) -> None:
        """Mount the public health check and the authenticated routes on `app`."""
        app.get("/healthz", tags=["meta"], summary="Health check")(self.healthz)

        api = APIRouter(dependencies=[Depends(authenticator)], responses=AUTH_RESPONSES)

        api.get("/models", tags=["meta"], summary="List checkpoints")(self.list_models)

        api.get("/presets", tags=["meta"], summary="List built-in question presets")(self.list_presets)

        api.post("/detect", tags=["meta"], summary="Detect script and language")(self.detect)

        api.post("/email/state", tags=["meta"], summary="Build a clean email state")(self.email_state_endpoint)

        api.post(
            "/predict",
            tags=["predict"],
            response_model_exclude_none=True,
            summary="Predict one state",
        )(self.predict)

        api.post(
            "/predict/bulk",
            tags=["predict"],
            response_model_exclude_none=True,
            summary="Predict many states",
        )(self.predict_bulk)

        api.post("/systemone", include_in_schema=False)(self.systemone_predict)

        api.post(
            "/v1/systemone",
            tags=["systemone", "predict"],
            summary="SystemOne / TypeSafe-compatible prediction",
        )(self.systemone_predict)

        app.include_router(api)

    def healthz(self) -> HealthResponse:
        """Report model readiness and configuration without requiring authentication."""
        return self._runtime.health()

    def list_models(self) -> ModelsResponse:
        """Available checkpoints and which are currently resident."""
        return self._runtime.models()

    def list_presets(self) -> dict[str, dict[str, Any]]:
        """Return the ready-to-use question sets (triage/email/guard/moderation/router)."""
        return self._backend.presets

    def detect(self, request: DetectRequest) -> dict[str, Any]:
        """Report script, best-effort language and `is_english` for a state (what routing uses)."""
        return cast(dict[str, Any], self._backend.library.detect_language(request.state))

    def email_state_endpoint(self, request: EmailStateRequest) -> dict[str, Any]:
        """Clean an email body and structure it as a state for `/predict`."""
        # The request fields mirror laya.email_state's parameters.
        return cast(dict[str, Any], self._backend.library.email_state(**request.model_dump()))

    def predict(self, request: PredictRequest) -> PredictResponse:
        """Run typed questions over a single state and return calibrated answers."""
        return self._predictions.predict(request)

    def predict_bulk(self, request: BulkPredictRequest) -> BulkPredictResponse:
        """Run typed questions over many states.

        Accepts either `states` with shared `questions`/`preset`, or `items` where each
        state can override its questions and model. Laya's public API predicts one state
        at a time (its internals can batch, but the public `Agent.predict` cannot), so
        this loops. Per-state errors are isolated and returned inline.
        """
        return self._predictions.predict_bulk(request)

    def systemone_predict(self, request: SystemOneRequest) -> SystemOneResponse:
        """Evaluate a SystemOne / TypeSafe-shaped request using a Laya checkpoint."""
        return self._systemone.predict(request)
