"""API key, bearer and HTTP Basic authentication; any configured method grants access."""

import base64
import secrets
from typing import Annotated

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from fastapi.security.http import HTTPBase

from app.config import Settings

# The schemes only extract credentials (auto_error=False) so OpenAPI documents them as
# alternatives; Authenticator decides. HTTPBase stands in for HTTPBasic, which decodes as
# ASCII and fails the request on a malformed header even when a valid API key is present.
API_KEY = APIKeyHeader(
    name="X-API-Key",
    scheme_name="ApiKeyAuth",
    description="A key from the server's API_KEYS.",
    auto_error=False,
)
BEARER = HTTPBearer(
    scheme_name="BearerAuth",
    description="An API key sent as a bearer token.",
    auto_error=False,
)
BASIC = HTTPBase(
    scheme="basic",
    scheme_name="BasicAuth",
    description="A user:password from the server's BASIC_AUTH.",
    auto_error=False,
)

Credentials = HTTPAuthorizationCredentials | None


class Authenticator:
    """FastAPI dependency that rejects requests no configured method accepts."""

    def __init__(self, settings: Settings) -> None:
        self._enabled = settings.auth_enabled
        self._keys = [key.encode() for key in settings.api_keys]
        self._accounts = [(user.encode(), password.encode()) for user, password in settings.basic_auth]

    def __call__(
        self,
        api_key: Annotated[str | None, Security(API_KEY)],
        bearer: Annotated[Credentials, Security(BEARER)],
        basic: Annotated[Credentials, Security(BASIC)],
    ) -> None:
        """Raise 401 unless a configured method accepts the request."""
        if not self.accepts(api_key, bearer, basic):
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing credentials",
                headers={"WWW-Authenticate": 'Basic realm="laya", Bearer'},
            )

    def accepts(self, api_key: str | None, bearer: Credentials, basic: Credentials) -> bool:
        """Return whether authentication is disabled or any supplied credential is valid."""
        return (
            not self._enabled
            or self._is_key(api_key)
            or (bearer is not None and self._is_key(bearer.credentials))
            or (basic is not None and self._is_account(basic))
        )

    def _is_key(self, token: str | None) -> bool:
        token = (token or "").strip()
        return bool(token) and any(secrets.compare_digest(token.encode(), key) for key in self._keys)

    def _is_account(self, basic: HTTPAuthorizationCredentials) -> bool:
        # HTTPBase accepts any scheme, so a Bearer header also arrives here.
        if basic.scheme.lower() != "basic":
            return False
        try:
            decoded = base64.b64decode(basic.credentials).decode()
        except ValueError:  # covers binascii.Error and UnicodeDecodeError
            return False
        user, _, password = (part.encode() for part in decoded.partition(":"))
        return any(
            secrets.compare_digest(user, expected_user) and secrets.compare_digest(password, expected_password)
            for expected_user, expected_password in self._accounts
        )
