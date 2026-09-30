"""Composition root: each created app gets its own router, lock and schema cache."""

from fastapi import FastAPI

from app.api import ApiRoutes
from app.authentication import Authenticator
from app.backend import LayaBackend
from app.config import Settings, load_local_environment
from app.services import ModelRuntime, PredictionService, SystemOneAdapter

DESCRIPTION = (
    "Run Laya typed decisions (choice / score / noul) over text, JSON objects "
    "or conversation turns, using your own questions or a built-in preset. "
    "Requests are auto-routed to the best checkpoint, or pinned with `model`.\n\n"
    "Authenticate with an API key (`X-API-Key` or `Authorization: Bearer`) or "
    "HTTP Basic, whichever is configured on the server."
)
TAGS = [
    {"name": "meta", "description": "Health, models, language detection and presets."},
    {"name": "predict", "description": "Typed decision prediction."},
]


class ApplicationFactory:
    """Builds FastAPI apps; settings default to the environment (and .env when present)."""

    def __init__(self, settings: Settings | None = None, backend: LayaBackend | None = None) -> None:
        if settings is None:
            load_local_environment()
            settings = Settings.from_environment()
        self._settings = settings
        self._backend = backend or LayaBackend()

    def create(self) -> FastAPI:
        """Create an app whose models load in its lifespan."""
        runtime = ModelRuntime(self._settings, self._backend)
        predictions = PredictionService(self._settings, runtime, self._backend)
        app = FastAPI(
            title="Laya API",
            version="0.6.1",
            description=DESCRIPTION,
            openapi_tags=TAGS,
            lifespan=runtime.lifespan,
        )
        routes = ApiRoutes(runtime, predictions, SystemOneAdapter(predictions), self._backend)
        routes.register(app, Authenticator(self._settings))
        return app
