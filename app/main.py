"""ASGI entry point: uvicorn app.main:app."""

from app.application import ApplicationFactory

app = ApplicationFactory().create()
