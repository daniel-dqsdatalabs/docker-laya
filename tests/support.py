"""Reusable test data and explicit dependency composition."""

from typing import Any
from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.application import ApplicationFactory
from app.backend import LayaBackend, PredictionRouter
from app.config import Settings
from app.schemas import NoulQuestion, Question
from app.services import ModelRuntime, PredictionService, SystemOneAdapter


class PredictionData:
    """Representative typed questions and model results."""

    @staticmethod
    def questions() -> dict[str, Question]:
        """Create a fresh typed refund question for tests that may mutate their inputs."""
        return {"refund": NoulQuestion(type="noul", instructions="Requesting a refund?")}

    @staticmethod
    def payload() -> dict[str, Any]:
        """Build the JSON-shaped request used to exercise HTTP validation."""
        return {
            "state": "Refund please",
            "questions": {key: value.model_dump() for key, value in PredictionData.questions().items()},
        }

    @staticmethod
    def result() -> dict[str, Any]:
        """Build a model result covering every answer type and extensible response fields."""
        return {
            "model": "english",
            "answers": {
                "refund": {
                    "type": "noul",
                    "noul": 0.9,
                    "confidence": 0.8,
                    "action": {"act_probability": 0.95},
                },
                "department": {
                    "type": "choice",
                    "choice": "billing",
                    "probabilities": {"billing": 1.0},
                    "confidence": 1.0,
                    "action": {"act_probability": 1.0},
                },
                "urgency": {
                    "type": "score",
                    "score": 1.0,
                    "legend": {"0": "low", "1": "high"},
                    "probabilities": {"low": 0.0, "high": 1.0},
                    "confidence": 1.0,
                    "action": {"act_probability": 1.0},
                },
            },
            "usage": {"input_tokens": 12, "output_tokens": 3},
            "routing": {"model": "english", "reason": "detected", "additional": True},
            "additional": "preserved",
        }


class ServiceContext:
    """Compose real services around a fake Laya boundary for focused tests."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Compose real services around mocks that never download or execute model checkpoints."""
        self.settings = settings if settings is not None else Settings()
        self.router = Mock(spec=PredictionRouter)
        self.router.loaded = ["english"]
        self.router.predict.return_value = PredictionData.result()
        self.backend = self._create_backend()
        self.runtime = ModelRuntime(self.settings, self.backend)
        self.predictions = PredictionService(self.settings, self.runtime, self.backend)
        self.systemone = SystemOneAdapter(self.predictions)

    def _create_backend(self) -> Mock:
        """Configure predictable preset, detection and email results at the Laya boundary."""
        backend = Mock(spec=LayaBackend)
        backend.create_router.return_value = self.router
        backend.presets = {
            name: {"preset": {"type": "noul", "instructions": name}}
            for name in ("triage", "email", "guard", "moderation", "router")
        }
        backend.library.detect_language.return_value = {"language": "en", "is_english": True}
        backend.library.email_state.return_value = {"subject": "Test", "body": "Cleaned"}
        return backend

    def client(self) -> TestClient:
        """Create an HTTP test client using the configured fake model backend."""
        return TestClient(ApplicationFactory(self.settings, self.backend).create())

    def running_services(self) -> TestClient:
        """Provide a client context that starts and stops the directly tested service runtime."""
        return TestClient(FastAPI(lifespan=self.runtime.lifespan))
