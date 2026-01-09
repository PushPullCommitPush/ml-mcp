"""Training backends for different frameworks and providers."""

from .base import TrainingBackend, BackendCapabilities
from .local import LocalBackend

__all__ = ["TrainingBackend", "BackendCapabilities", "LocalBackend"]
