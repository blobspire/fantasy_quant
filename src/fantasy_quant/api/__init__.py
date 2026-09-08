"""The local HTTP surface: JSON out of `report.py`, and nothing computed on the way.

Two modules. `cache.py` holds answers with their ages and an explicit invalidate;
`server.py` is the FastAPI app, the `report.Workspace` per league, and `fq dash`.

Imports are lazy on purpose. `fastapi` and `uvicorn` are dependencies of the dashboard
and not of the analytics, and `from fantasy_quant import api` must not drag a web
framework into `fq odds` or into the test suite of a module that has nothing to do with
HTTP. `api.server` is imported when something asks for it, which is either `fq dash` or
`uvicorn fantasy_quant.api.server:app`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .server import Engine, Settings, app, app_cli, create_app

__all__ = ["Engine", "Settings", "app", "app_cli", "create_app"]


def __getattr__(name: str) -> Any:
    """Resolve `api.create_app` and friends without importing FastAPI at package import."""
    if name in __all__:
        from . import server

        return getattr(server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
