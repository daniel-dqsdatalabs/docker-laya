"""Immutable configuration loaded once per application instance."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

AVAILABLE_MODELS = ("english", "multilingual", "typed-decisions")
DEFAULT_MODEL_ID = "convaiinnovations/laya"
ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def load_local_environment() -> None:
    """Load the project dotenv file and normalize the Hugging Face token.

    Existing process variables win over the file. A blank token is removed so Hub
    clients stay anonymous instead of sending an empty bearer credential.
    """
    load_dotenv(ENV_FILE, override=False)
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        os.environ["HF_TOKEN"] = token
        return
    os.environ.pop("HF_TOKEN", None)


@dataclass(frozen=True)
class Settings:
    """Environment settings with the same defaults as the public Docker image."""

    api_keys: tuple[str, ...] = ()
    basic_auth: tuple[tuple[str, str], ...] = ()
    max_bulk_items: int | None = None
    models: tuple[str, ...] = ("english",)
    model_id: str = DEFAULT_MODEL_ID
    model_subfolder: str | None = None
    device: str = "cpu"

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> Settings:
        """Read deployment settings once, or parse an explicitly supplied environment mapping."""
        values = os.environ if environment is None else environment
        limit = values.get("MAX_BULK_ITEMS", "").strip()
        return cls(
            api_keys=cls._split_values(values.get("API_KEYS", "")),
            basic_auth=cls._parse_basic_auth(values.get("BASIC_AUTH", "")),
            max_bulk_items=int(limit) if limit and limit != "0" else None,
            models=cls._split_values(values.get("MODELS", "english")),
            model_id=values.get("MODEL_ID", DEFAULT_MODEL_ID),
            model_subfolder=values.get("MODEL_SUBFOLDER") or None,
            device=values.get("DEVICE", "cpu"),
        )

    @staticmethod
    def _split_values(value: str) -> tuple[str, ...]:
        """Trim comma-separated values and discard empty entries."""
        return tuple(part.strip() for part in value.split(",") if part.strip())

    @staticmethod
    def _parse_basic_auth(value: str) -> tuple[tuple[str, str], ...]:
        """Parse account pairs while preserving colons inside passwords."""
        accounts = []
        for pair in value.split(","):
            if ":" in pair:
                username, password = pair.split(":", 1)
                accounts.append((username.strip(), password.strip()))
        return tuple(accounts)

    @property
    def auth_enabled(self) -> bool:
        """Indicate whether at least one authentication method is configured."""
        return bool(self.api_keys or self.basic_auth)

    @property
    def auth_methods(self) -> list[str]:
        """List enabled authentication methods in the order exposed by the health endpoint."""
        methods = (("apikey", self.api_keys), ("basic", self.basic_auth))
        return [name for name, credentials in methods if credentials]
