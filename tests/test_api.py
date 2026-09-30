"""HTTP regression tests, including the complete published OpenAPI contract."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock, call, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.application import ApplicationFactory
from app.config import Settings
from tests.support import PredictionData, ServiceContext


class TestHttpContracts:
    """Requests exercise real routing, auth dependencies, validation and serialization."""

    def test_health_is_public_and_reports_unstarted_runtime(self) -> None:
        """Keep health accessible without authentication before model startup."""
        client = ServiceContext(Settings(api_keys=("secret",))).client()
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {
            "status": "ok",
            "model_loaded": False,
            "device": "cpu",
            "auth_enabled": True,
            "auth_methods": ["apikey"],
            "models": [],
            "available_models": ["english", "multilingual", "typed-decisions"],
        }

    @pytest.mark.parametrize(
        "method, path, body",
        [
            ("GET", "/models", None),
            ("GET", "/presets", None),
            ("POST", "/detect", {"state": "hello"}),
            ("POST", "/email/state", {"body": "hello"}),
            ("POST", "/predict", PredictionData.payload()),
            ("POST", "/predict/bulk", {"states": ["hello"], "preset": "triage"}),
            ("POST", "/systemone", {**PredictionData.payload(), "model": "english"}),
            ("POST", "/v1/systemone", {**PredictionData.payload(), "model": "english"}),
        ],
    )
    def test_auth_guards_every_protected_route(self, method: str, path: str, body: dict[str, Any] | None) -> None:
        """Reject unauthorized requests before any protected operation reaches its backend."""
        context = ServiceContext(Settings(api_keys=("secret",)))
        with context.client() as client:
            response = client.request(method, path, json=body)
        assert response.status_code == 401
        assert response.json() == {"detail": "Invalid or missing credentials"}
        assert response.headers["www-authenticate"] == 'Basic realm="laya", Bearer'
        context.router.predict.assert_not_called()
        context.backend.library.detect_language.assert_not_called()

    @pytest.mark.parametrize(
        "headers",
        [
            {"X-API-Key": "secret"},
            {"Authorization": "Bearer secret"},
            {"Authorization": "Basic dXNlcjpwYXNz"},
        ],
    )
    def test_http_authentication_alternatives(self, headers: dict[str, str]) -> None:
        """Exercise API key, bearer and Basic authentication through real HTTP dependencies."""
        context = ServiceContext(Settings(api_keys=("secret",), basic_auth=(("user", "pass"),)))
        with context.client() as client:
            response = client.get("/models", headers=headers)
        assert response.status_code == 200
        assert response.json()["loaded"] == ["english"]

    def test_single_prediction_and_optional_fields(self) -> None:
        """Preserve response extensions and omit absent optional routing metadata."""
        context = ServiceContext()
        context.router.predict.return_value.pop("routing")
        with context.client() as client:
            response = client.post("/predict", json=PredictionData.payload())
        assert response.status_code == 200
        assert response.json() == context.router.predict.return_value
        assert "routing" not in response.json()

    def test_bulk_excludes_null_results_but_returns_item_errors(self) -> None:
        """Serialize item failures without null prediction fields or losing later successes."""
        context = ServiceContext()
        context.router.predict.side_effect = [
            RuntimeError("invalid state"),
            PredictionData.result(),
        ]
        with context.client() as client:
            response = client.post("/predict/bulk", json={"states": ["bad", "good"], "preset": "triage"})
        assert response.status_code == 200
        assert response.json()["results"][0] == {"ok": False, "error": "invalid state"}
        assert "error" not in response.json()["results"][1]
        assert response.json()["count"] == 2

    def test_bulk_limit_response(self) -> None:
        """Return the established HTTP status and message for an oversized batch."""
        with ServiceContext(Settings(max_bulk_items=1)).client() as client:
            response = client.post("/predict/bulk", json={"states": ["one", "two"], "preset": "triage"})
        assert response.status_code == 422
        assert response.json() == {"detail": "Too many states: 2 > MAX_BULK_ITEMS=1"}

    @pytest.mark.parametrize("path", ["/systemone", "/v1/systemone"])
    def test_systemone_aliases(self, path: str) -> None:
        """Serve the same compatible response shape through both SystemOne URLs."""
        with ServiceContext().client() as client:
            response = client.post(path, json={**PredictionData.payload(), "model": "typesafe/laya-english"})
        assert response.status_code == 200
        assert response.json()["model"] == "typesafe/laya-english"
        assert response.json()["answers"]["refund"] == {"type": "noul", "noul": 0.9}
        assert response.json()["provider"] == "Laya"

    def test_metadata_helpers(self) -> None:
        """Expose presets and forward language and email requests to the configured backend."""
        context = ServiceContext()
        with context.client() as client:
            assert client.get("/presets").json() == context.backend.presets
            detected = client.post("/detect", json={"state": {"text": "Olá"}})
            cleaned = client.post("/email/state", json={"body": "Quoted", "sender": "a@b.test", "clean": False})
        library = context.backend.library
        assert detected.json() == library.detect_language.return_value
        assert cleaned.json() == library.email_state.return_value
        library.detect_language.assert_called_once_with({"text": "Olá"})
        library.email_state.assert_called_once_with(subject="", body="Quoted", sender="a@b.test", clean=False)

    @pytest.mark.parametrize(
        "method, path, body",
        [
            ("GET", "/models", None),
            ("POST", "/predict", PredictionData.payload()),
            ("POST", "/predict/bulk", {"states": ["one"], "preset": "triage"}),
            ("POST", "/systemone", {**PredictionData.payload(), "model": "english"}),
        ],
    )
    def test_unavailable_model_response(self, method: str, path: str, body: dict[str, Any] | None) -> None:
        """Return the documented readiness error for routes that require loaded checkpoints."""
        response = ServiceContext().client().request(method, path, json=body)
        assert response.status_code == 503
        assert response.json() == {"detail": "Model not loaded yet"}


class TestRequestValidation:
    """Pydantic rejects invalid shapes before they reach inference."""

    @pytest.mark.parametrize(
        "path, payload",
        [
            ("/predict", {"state": "text"}),
            ("/predict", {**PredictionData.payload(), "preset": "triage"}),
            ("/predict", {"state": "text", "preset": "unknown"}),
            ("/predict", {**PredictionData.payload(), "model": "unknown"}),
            (
                "/predict",
                {"state": "text", "questions": {"q": {"type": "unknown", "instructions": "?"}}},
            ),
            ("/predict/bulk", {}),
            ("/predict/bulk", {"states": [], "preset": "triage"}),
            ("/predict/bulk", {"items": []}),
            ("/predict/bulk", {"states": ["a"]}),
            ("/predict/bulk", {"states": ["a"], "items": [{"state": "b"}], "preset": "triage"}),
            ("/predict/bulk", {"states": ["a"], "questions": {}, "preset": "triage"}),
            ("/v1/systemone", PredictionData.payload()),
        ],
    )
    def test_invalid_requests(self, path: str, payload: dict[str, Any]) -> None:
        """Reject invalid request shapes before any model inference is attempted."""
        context = ServiceContext()
        with context.client() as client:
            response = client.post(path, json=payload)
        assert response.status_code == 422
        context.router.predict.assert_not_called()

    @pytest.mark.parametrize("state", ["text", {"subject": "test"}, [{"role": "user", "content": "hello"}]])
    def test_supported_state_shapes(self, state: Any) -> None:
        """Accept text, objects and conversation arrays without altering their contents."""
        context = ServiceContext()
        with context.client() as client:
            response = client.post("/predict", json={"state": state, "questions": {}})
        assert response.status_code == 200
        assert context.router.predict.call_args.args[0] == state


class TestApplicationFactory:
    """Applications do not share router state, credentials or cached schemas."""

    def test_startup_loads_dotenv_before_reading_the_environment(self) -> None:
        """Expose HF_TOKEN before settings and the model router are created."""
        startup = Mock()
        startup.read.return_value = Settings()
        with (
            patch("app.application.load_local_environment", startup.load),
            patch("app.application.Settings.from_environment", startup.read),
        ):
            ApplicationFactory()
        assert startup.mock_calls == [call.load(), call.read()]

    def test_injected_settings_skip_dotenv(self) -> None:
        """Leave explicit test configuration independent of the local dotenv file."""
        with patch("app.application.load_local_environment") as loader:
            ApplicationFactory(Settings())
        loader.assert_not_called()

    def test_instances_are_independent(self) -> None:
        """Keep credentials, schema caches and router lifetimes separate across applications."""
        first = ServiceContext(Settings(api_keys=("first",)))
        second = ServiceContext(Settings(api_keys=("second",)))
        with first.client() as first_client, second.client() as second_client:
            assert first_client.get("/models", headers={"X-API-Key": "first"}).status_code == 200
            assert second_client.get("/models", headers={"X-API-Key": "first"}).status_code == 401
            assert isinstance(first_client.app, FastAPI)
            assert isinstance(second_client.app, FastAPI)
            assert first_client.app.openapi() is not second_client.app.openapi()
        assert not first_client.get("/healthz").json()["model_loaded"]

    def test_shutdown_of_one_app_does_not_clear_another(self) -> None:
        """Keep one application ready when another created by the same factory shuts down."""
        factory = ApplicationFactory(Settings(), ServiceContext().backend)
        with TestClient(factory.create()) as first:
            with TestClient(factory.create()) as second:
                assert second.get("/healthz").json()["model_loaded"]
            assert first.get("/healthz").json()["model_loaded"]

    def test_entire_openapi_contract_is_preserved_without_laya(self) -> None:
        """Match the published schema without importing the optional model runtime."""
        expected = json.loads((Path(__file__).parents[1] / "openapi.json").read_text())
        with patch("app.backend.import_module", side_effect=AssertionError("Must not import Laya")):
            app = ApplicationFactory(Settings()).create()
            assert app.openapi() == expected
            assert app.openapi() is app.openapi_schema
            assert TestClient(app).get("/openapi.json").json() == expected
