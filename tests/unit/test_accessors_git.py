# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Unit tests for GitAccessor."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from openviking.parse.accessors import GitAccessor
from openviking.utils import code_hosting_utils


def _mock_config():
    return SimpleNamespace(
        code=SimpleNamespace(
            github_domains=["github.com", "www.github.com"],
            gitlab_domains=["gitlab.com", "www.gitlab.com"],
            code_hosting_domains=["github.com", "gitlab.com"],
        )
    )


@pytest.fixture(autouse=True)
def _patch_config():
    with patch.object(code_hosting_utils, "get_openviking_config", side_effect=_mock_config):
        yield


def _mock_config():
    return SimpleNamespace(
        code=SimpleNamespace(
            github_domains=["github.com", "www.github.com"],
            gitlab_domains=["gitlab.com", "www.gitlab.com"],
            bitbucket_domains=["bitbucket.org", "www.bitbucket.org"],
            azure_devops_domains=[
                "dev.azure.com",
                "ssh.dev.azure.com",
                "vs-ssh.visualstudio.com",
            ],
            code_hosting_domains=["github.com", "gitlab.com"],
        )
    )


class TestGitAccessor:
    """Tests for GitAccessor."""

    @pytest.fixture(autouse=True)
    def _patch_config(self):
        with patch(
            "openviking_cli.utils.config.open_viking_config.OpenVikingConfigSingleton.get_instance",
            side_effect=_mock_config,
        ):
            yield

    @pytest.fixture
    def accessor(self) -> GitAccessor:
        """Create a GitAccessor instance."""
        return GitAccessor()

    def test_priority(self, accessor: GitAccessor) -> None:
        """GitAccessor should have correct priority."""
        assert accessor.priority == 80

    @pytest.mark.parametrize(
        "source",
        [
            "git@github.com:volcengine/OpenViking.git",
            "git@gitlab.com:org/repo.git",
            "git@ssh.dev.azure.com:v3/org/project/repo",
            "ssh://git@ssh.dev.azure.com/v3/org/project/repo.git",
            "git@vs-ssh.visualstudio.com:v3/org/project/repo",
        ],
    )
    def test_can_handle_git_ssh_url(self, accessor: GitAccessor, source: str) -> None:
        """GitAccessor should handle git@ SSH URLs."""
        assert accessor.can_handle(source) is True

    @pytest.mark.parametrize(
        "source",
        [
            "https://github.com/volcengine/OpenViking",
            "https://github.com/volcengine/OpenViking.git",
            "https://gitlab.com/org/repo",
            "http://github.com/org/repo",
        ],
    )
    def test_can_handle_github_http_url(self, accessor: GitAccessor, source: str) -> None:
        """GitAccessor should handle GitHub/GitLab HTTP URLs."""
        assert accessor.can_handle(source) is True

    @pytest.mark.parametrize(
        "source",
        [
            "https://github.com/volcengine/OpenViking/tree/main",
            "https://github.com/volcengine/OpenViking/tree/abc1234",
        ],
    )
    def test_can_handle_github_with_ref(self, accessor: GitAccessor, source: str) -> None:
        """GitAccessor should handle GitHub URLs with branch/commit."""
        assert accessor.can_handle(source) is True

    @pytest.mark.parametrize(
        "source",
        [
            "https://dev.azure.com/org/project/_git/repo",
            "https://dev.azure.com/org/project/_git/repo.git",
        ],
    )
    def test_can_handle_azure_devops_http_url(self, accessor: GitAccessor, source: str) -> None:
        """GitAccessor should handle Azure DevOps repository URLs."""
        assert accessor.can_handle(source) is True

    def test_can_handle_git_protocol_url(self, accessor: GitAccessor) -> None:
        """GitAccessor should handle git:// URLs."""
        assert accessor.can_handle("git://github.com/volcengine/OpenViking.git") is True

    def test_normalize_repo_url_ssh_with_userinfo_and_ref(self, accessor: GitAccessor) -> None:
        """GitAccessor should normalize ssh URLs with userinfo using the shared host matcher."""
        assert (
            accessor._normalize_repo_url("ssh://git@github.com:443/volcengine/OpenViking/tree/main")
            == "ssh://git@github.com:443/volcengine/OpenViking"
        )

    @pytest.mark.parametrize(
        "source",
        [
            "/path/to/repo.git",
        ],
    )
    def test_can_handle_local_files(self, accessor: GitAccessor, source: str) -> None:
        """GitAccessor should handle local .git files."""
        assert accessor.can_handle(Path(source)) is True

    def test_cannot_handle_local_zip_file(self, accessor: GitAccessor) -> None:
        """GitAccessor should leave local zip files to LocalAccessor/ZipParser."""
        assert accessor.can_handle(Path("/path/to/archive.zip")) is False

    @pytest.mark.parametrize(
        "source",
        [
            "https://example.com/page.html",
            "https://github.com/volcengine/OpenViking/issues/123",
            "https://dev.azure.com/org/project/_build",
            "https://dev.azure.com/org/project/_git/repo?path=/README.md",
            "https://dev.azure.com/org/project/_git/repo/pullrequest/123",
            "https://dev.azure.com/org/project/_git/repo/commit/abc1234",
            "git@example.com:repo",
        ],
    )
    def test_cannot_handle_other_urls(self, accessor: GitAccessor, source: str) -> None:
        """GitAccessor should not handle non-git URLs or files."""
        assert accessor.can_handle(source) is False

    @pytest.mark.parametrize(
        "source",
        [
            "https://bitbucket.org/myworkspace/myrepo",
            "https://bitbucket.org/myworkspace/myrepo.git",
            "git@bitbucket.org:myworkspace/myrepo.git",
        ],
    )
    def test_can_handle_bitbucket_url(self, accessor: GitAccessor, source: str) -> None:
        """GitAccessor should handle Bitbucket repository URLs."""
        assert accessor.can_handle(source) is True

    # --- Per-request token: ZIP header construction ---

    def test_zip_headers_github_uses_per_request_token(self, accessor: GitAccessor) -> None:
        """The GitHub codeload archive API uses the REST ``token`` scheme (the
        known-working form), not the Basic smart-HTTP form used for git clone."""
        headers = accessor._zip_headers("https://github.com/o/r", "TOK")
        assert headers["Authorization"] == "token TOK"
        assert headers["User-Agent"] == "OpenViking"

    def test_zip_headers_no_token_means_no_authorization(
        self, accessor: GitAccessor, monkeypatch
    ) -> None:
        """With no per-request token, no Authorization header is sent -- even if the
        legacy GITHUB_TOKEN env var is set (the env auth has been removed)."""
        monkeypatch.setenv("GITHUB_TOKEN", "legacy-env-token")
        headers = accessor._zip_headers("https://github.com/o/r", None)
        assert "Authorization" not in headers

    # --- Per-request token: clone uses env, never argv ---

    @pytest.mark.asyncio
    async def test_git_clone_passes_token_via_env_not_argv(
        self, accessor: GitAccessor, tmp_path, monkeypatch
    ) -> None:
        """The token must reach git via GIT_CONFIG_* env vars, never via argv
        (argv leaks into /proc and into _run_git's error string)."""
        captured: dict = {}

        async def _fake_run_git(args, cwd=None, env=None):
            captured["args"] = args
            captured["env"] = env
            return ""

        monkeypatch.setattr(accessor, "_run_git", _fake_run_git)

        await accessor._git_clone(
            "https://github.com/o/r",
            str(tmp_path),
            git_auth_token="SUPERSECRET",
        )

        import base64

        assert "SUPERSECRET" not in " ".join(captured["args"])
        env = captured["env"]
        assert env is not None
        assert env["GIT_CONFIG_KEY_0"] == "http.https://github.com/.extraHeader"
        expected = "Authorization: Basic " + base64.b64encode(
            b"x-access-token:SUPERSECRET"
        ).decode()
        assert env["GIT_CONFIG_VALUE_0"] == expected
        # The raw token is base64-wrapped, never in cleartext on the command line.
        assert "SUPERSECRET" not in env["GIT_CONFIG_KEY_0"]

    # --- access() routes the token to the right downloader ---

    @pytest.mark.asyncio
    async def test_access_github_threads_token_to_zip_download(
        self, accessor: GitAccessor, monkeypatch
    ) -> None:
        captured: dict = {}

        async def _fake_zip(repo_url, branch, target_dir, git_auth_token=None):
            captured["token"] = git_auth_token
            content = Path(target_dir) / "content"
            content.mkdir()
            return content, "o/r"

        monkeypatch.setattr(accessor, "_github_zip_download", _fake_zip)
        res = await accessor.access("https://github.com/o/r", git_auth_token="TOK")
        try:
            assert captured["token"] == "TOK"
        finally:
            res.cleanup()

    @pytest.mark.asyncio
    async def test_access_bitbucket_uses_authenticated_clone(
        self, accessor: GitAccessor, monkeypatch
    ) -> None:
        captured: dict = {}

        async def _fake_clone(url, target_dir, branch=None, commit=None, git_auth_token=None):
            captured["url"] = url
            captured["token"] = git_auth_token
            return "ws/repo"

        monkeypatch.setattr(accessor, "_git_clone", _fake_clone)
        res = await accessor.access("https://bitbucket.org/ws/repo", git_auth_token="TOK")
        try:
            assert captured["url"] == "https://bitbucket.org/ws/repo"
            assert captured["token"] == "TOK"
        finally:
            res.cleanup()

    @pytest.mark.asyncio
    async def test_git_clone_without_token_passes_no_git_config_env(
        self, accessor: GitAccessor, tmp_path, monkeypatch
    ) -> None:
        """No token -> no injected git config env (env stays None)."""
        captured: dict = {}

        async def _fake_run_git(args, cwd=None, env=None):
            captured["env"] = env
            return ""

        monkeypatch.setattr(accessor, "_run_git", _fake_run_git)

        await accessor._git_clone("https://github.com/o/r", str(tmp_path))

        assert captured["env"] is None
