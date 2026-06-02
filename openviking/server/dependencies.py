# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Dependency injection for OpenViking HTTP Server."""

from typing import TYPE_CHECKING, Optional

from fastapi import Depends

from openviking.server.auth import get_request_context
from openviking.server.identity import RequestContext
from openviking.server.project_init import ProjectInitGuard, build_default_initializer
from openviking.service.core import OpenVikingService

if TYPE_CHECKING:
    from openviking.server.config import ServerConfig

_service: Optional[OpenVikingService] = None
_server_config: Optional["ServerConfig"] = None
_PROJECT_GUARD: Optional[ProjectInitGuard] = None


def get_service() -> OpenVikingService:
    """Get the OpenVikingService instance.

    Returns:
        OpenVikingService instance

    Raises:
        RuntimeError: If service is not initialized
    """
    if _service is None:
        raise RuntimeError("OpenVikingService not initialized")
    return _service


def set_service(service: OpenVikingService) -> None:
    """Set the OpenVikingService instance.

    Args:
        service: OpenVikingService instance to set
    """
    global _service, _PROJECT_GUARD
    _service = service
    _PROJECT_GUARD = None


def get_server_config() -> Optional["ServerConfig"]:
    """Return the active ServerConfig if one was registered, else None.

    MCP tools that need server-level settings (e.g. public_base_url for upload URLs)
    use this to read the loaded ServerConfig without going through ``app.state``.
    """
    return _server_config


def set_server_config(config: "ServerConfig") -> None:
    """Register the active ServerConfig at server bootstrap."""
    global _server_config
    _server_config = config


def get_project_guard() -> ProjectInitGuard:
    """Return the process-wide lazy per-project initialization guard."""
    global _PROJECT_GUARD
    if _PROJECT_GUARD is None:
        _PROJECT_GUARD = ProjectInitGuard(build_default_initializer(get_service()))
    return _PROJECT_GUARD


async def ensure_project_ready(
    ctx: RequestContext = Depends(get_request_context),
) -> RequestContext:
    """Data-plane dependency: lazily bootstrap the request's project, then
    return the resolved RequestContext (drop-in for ``get_request_context``)."""
    await get_project_guard().ensure(ctx)
    return ctx
