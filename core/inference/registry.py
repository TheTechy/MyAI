"""
core/inference/registry.py — backend registry / factory (MyAI v2, F1 Story 3 Task 1)

The SINGLE place that knows which backend kinds exist. config.py is deliberately
ignorant of this: it just reads the `kind` string, so adding a backend touches
this file and nothing else.

    from core.inference.config import load_config
    from core.inference.registry import create_backend

    backend = create_backend(load_config())   # not yet loaded
    backend.load()

Adding the deferred SME/HTTP backend (F6) is then three lines here — an import
and a register() call — plus the backend module itself. Nothing in the core
changes, which is the seam doing its job.

Import-light on purpose: this module pulls in no llama_cpp (the in-process
backend imports it lazily inside load()), so importing the registry costs
nothing and works on a machine with no inference libraries installed.
"""
from __future__ import annotations

from typing import Dict, List, Type

from .base import InferenceBackend
from .config import BackendConfig
from .llama_cpp_backend import LlamaCppBackend


class BackendNotFoundError(KeyError):
    """Raised when a config names a backend kind that isn't registered."""

    def __init__(self, kind: str, available: List[str]) -> None:
        self.kind = kind
        self.available = available
        super().__init__(
            f"unknown backend kind {kind!r} — available: {', '.join(available)}. "
            f"Set MYAI_BACKEND to one of those, or leave it unset for the "
            f"in-process default."
        )

    def __str__(self) -> str:          # KeyError repr()s its arg; keep it readable
        return self.args[0]


# kind → backend class. The one slot filled today is the in-process default.
#
# RESERVED, NOT IMPLEMENTED — "http": the SME hybrid cloud-offload backend
# (ADR-0001, F6). It speaks this same OpenAI-shaped contract over real HTTP to a
# local server or cloud endpoint. It is deliberately absent rather than stubbed:
# a stub that imports would let a typo silently select a non-functional backend,
# whereas an absent key fails loudly with the list of what actually works.
_BACKENDS: Dict[str, Type[InferenceBackend]] = {
    "llama_cpp": LlamaCppBackend,
}


def register(kind: str, cls: Type[InferenceBackend], *, replace: bool = False) -> None:
    """Register a backend class under `kind`.

    Rejects anything that isn't an InferenceBackend subclass — the seam is only
    worth having if membership is actually enforced. Rejects duplicate kinds
    unless `replace=True`, so an accidental second registration is caught rather
    than silently shadowing the first (tests legitimately want replace=True).
    """
    if not isinstance(cls, type) or not issubclass(cls, InferenceBackend):
        raise TypeError(
            f"{getattr(cls, '__name__', cls)!r} must be a subclass of InferenceBackend"
        )
    if kind in _BACKENDS and not replace:
        raise ValueError(
            f"backend kind {kind!r} is already registered to "
            f"{_BACKENDS[kind].__name__}; pass replace=True to override"
        )
    _BACKENDS[kind] = cls


def unregister(kind: str) -> None:
    """Remove a registered kind. Primarily for tests."""
    _BACKENDS.pop(kind, None)


def available_kinds() -> List[str]:
    """Registered backend kinds, sorted."""
    return sorted(_BACKENDS)


def get_backend_class(kind: str) -> Type[InferenceBackend]:
    """Look up a backend class, raising BackendNotFoundError if absent."""
    try:
        return _BACKENDS[kind]
    except KeyError:
        raise BackendNotFoundError(kind, available_kinds()) from None


def create_backend(config: BackendConfig) -> InferenceBackend:
    """Build the backend named by `config.kind`.

    Returns it UNLOADED — loading weights is slow and sometimes the caller only
    wants to inspect or swap. Call load() explicitly.
    """
    return get_backend_class(config.kind)(config)
