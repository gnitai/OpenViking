# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Unit tests for per-request git auth helpers."""

import base64

from openviking.utils.git_auth import (
    archive_auth_header_value,
    auth_header_value,
    git_config_env,
)


def _basic(userinfo: str) -> str:
    return "Basic " + base64.b64encode(userinfo.encode()).decode()


class TestAuthHeaderValue:
    def test_github_uses_basic_x_access_token(self) -> None:
        assert auth_header_value("github.com", "TOK") == _basic("x-access-token:TOK")

    def test_github_www_host(self) -> None:
        assert auth_header_value("www.github.com", "TOK") == _basic("x-access-token:TOK")

    def test_bitbucket_uses_basic_x_token_auth(self) -> None:
        assert auth_header_value("bitbucket.org", "TOK") == _basic("x-token-auth:TOK")

    def test_gitlab_uses_bearer(self) -> None:
        assert auth_header_value("gitlab.com", "TOK") == "Bearer TOK"

    def test_unknown_host_defaults_to_bearer(self) -> None:
        assert auth_header_value("git.example.com", "TOK") == "Bearer TOK"


class TestArchiveAuthHeaderValue:
    """The HTTP archive/codeload API expects a different scheme than git transport.

    GitHub codeload accepts the REST ``token <token>`` form (the form the original
    env-based code used); it does NOT speak the Basic ``x-access-token`` smart-HTTP
    form used for git clone. Keeping these separate avoids a silent 401 -> clone
    fallback that would defeat the ZIP fast-path.
    """

    def test_github_uses_token_scheme(self) -> None:
        assert archive_auth_header_value("github.com", "TOK") == "token TOK"

    def test_gitlab_uses_bearer(self) -> None:
        assert archive_auth_header_value("gitlab.com", "TOK") == "Bearer TOK"

    def test_unknown_host_defaults_to_bearer(self) -> None:
        assert archive_auth_header_value("git.example.com", "TOK") == "Bearer TOK"


class TestGitConfigEnv:
    def test_scopes_extraheader_to_repo_base_url(self) -> None:
        env = git_config_env("https://github.com/owner/repo", "TOK")
        assert env["GIT_CONFIG_COUNT"] == "1"
        # Host-scoped, NOT a bare http.extraHeader (which would leak to submodule hosts).
        assert env["GIT_CONFIG_KEY_0"] == "http.https://github.com/.extraHeader"
        assert env["GIT_CONFIG_VALUE_0"] == f"Authorization: {_basic('x-access-token:TOK')}"

    def test_bitbucket_base_url_and_scheme(self) -> None:
        env = git_config_env("https://bitbucket.org/ws/repo.git", "TOK")
        assert env["GIT_CONFIG_KEY_0"] == "http.https://bitbucket.org/.extraHeader"
        assert env["GIT_CONFIG_VALUE_0"] == f"Authorization: {_basic('x-token-auth:TOK')}"

    def test_raw_token_never_appears_in_config_key(self) -> None:
        env = git_config_env("https://github.com/owner/repo", "SUPERSECRET")
        assert "SUPERSECRET" not in env["GIT_CONFIG_KEY_0"]
        assert "SUPERSECRET" not in env["GIT_CONFIG_COUNT"]
