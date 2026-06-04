# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Per-request git authentication helpers.

These helpers turn a transient, per-request access token (OAuth token or PAT,
resolved by the caller for a single ingest) into the HTTP ``Authorization`` value
or scoped git config required to fetch a private repository. The token is used
transiently and is never persisted or logged.

The auth scheme is provider-specific (a protocol detail of each host), not a
deployment setting, so it is resolved by hostname here rather than from config.
"""

from __future__ import annotations

import base64
from urllib.parse import urlparse

_GITHUB_HOSTS = ("github.com",)
_BITBUCKET_HOSTS = ("bitbucket.org",)
_GITLAB_HOSTS = ("gitlab.com",)


def _host_matches(host: str, domains: tuple[str, ...]) -> bool:
    host = host.lower()
    return any(host == d or host.endswith("." + d) for d in domains)


def _basic(userinfo: str) -> str:
    return "Basic " + base64.b64encode(userinfo.encode()).decode()


def auth_header_value(host: str, token: str) -> str:
    """Return the HTTP ``Authorization`` header value for ``host`` and ``token``.

    - GitHub: Basic ``x-access-token:<token>`` (works for OAuth tokens and PATs)
    - Bitbucket Cloud: Basic ``x-token-auth:<token>`` (OAuth / access tokens)
    - GitLab and unknown/self-hosted hosts: Bearer ``<token>``
    """
    if _host_matches(host, _GITHUB_HOSTS):
        return _basic(f"x-access-token:{token}")
    if _host_matches(host, _BITBUCKET_HOSTS):
        return _basic(f"x-token-auth:{token}")
    return f"Bearer {token}"


def archive_auth_header_value(host: str, token: str) -> str:
    """Return the ``Authorization`` value for an archive/codeload HTTP download.

    This is a *different* context from git transport (:func:`auth_header_value`):
    the HTTP archive APIs speak their provider's REST auth scheme, not the
    smart-HTTP Basic form. Using the wrong scheme here yields a 401 that silently
    falls back to ``git clone``, defeating the ZIP fast-path.

    - GitHub codeload: ``token <token>`` (the form the original code used)
    - GitLab and unknown hosts: Bearer ``<token>``
    """
    if _host_matches(host, _GITHUB_HOSTS):
        return f"token {token}"
    return f"Bearer {token}"


def git_config_env(repo_url: str, token: str) -> dict[str, str]:
    """Build ``GIT_CONFIG_*`` env vars injecting a scoped auth header for a clone.

    Using ``GIT_CONFIG_COUNT``/``GIT_CONFIG_KEY_*``/``GIT_CONFIG_VALUE_*`` keeps the
    token out of argv (it never reaches ``/proc/<pid>/cmdline`` or any error string
    that echoes the git command). The ``http.<base-url>.extraHeader`` key is scoped
    to the repository's host so a ``--recursive`` submodule fetch cannot leak the
    token to a third-party submodule host. Nothing is written to ``.git/config``.
    """
    parsed = urlparse(repo_url)
    host = parsed.hostname or ""
    base_url = f"{parsed.scheme}://{host}/"
    return {
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": f"http.{base_url}.extraHeader",
        "GIT_CONFIG_VALUE_0": f"Authorization: {auth_header_value(host, token)}",
    }
