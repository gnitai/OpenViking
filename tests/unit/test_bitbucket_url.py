# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Unit tests for Bitbucket code-hosting URL support."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from openviking.utils import code_hosting_utils
from openviking.utils.code_hosting_utils import (
    is_bitbucket_url,
    is_code_hosting_url,
    parse_code_hosting_url,
)


def _mock_config():
    return SimpleNamespace(
        code=SimpleNamespace(
            github_domains=["github.com", "www.github.com"],
            gitlab_domains=["gitlab.com", "www.gitlab.com"],
            bitbucket_domains=["bitbucket.org", "www.bitbucket.org"],
            azure_devops_domains=["dev.azure.com"],
            code_hosting_domains=["github.com", "gitlab.com"],
        )
    )


@pytest.fixture(autouse=True)
def _patch_config():
    with patch.object(code_hosting_utils, "get_openviking_config", side_effect=_mock_config):
        yield


class TestBitbucketDetection:
    def test_is_bitbucket_url_true(self) -> None:
        assert is_bitbucket_url("https://bitbucket.org/myworkspace/myrepo") is True

    def test_is_bitbucket_url_ssh(self) -> None:
        assert is_bitbucket_url("git@bitbucket.org:myworkspace/myrepo.git") is True

    def test_is_bitbucket_url_false_for_github(self) -> None:
        assert is_bitbucket_url("https://github.com/owner/repo") is False

    def test_bitbucket_counts_as_code_hosting(self) -> None:
        assert is_code_hosting_url("https://bitbucket.org/ws/repo") is True

    def test_parse_extracts_workspace_repo(self) -> None:
        assert parse_code_hosting_url("https://bitbucket.org/ws/repo") == "ws/repo"

    def test_parse_strips_git_suffix(self) -> None:
        assert parse_code_hosting_url("https://bitbucket.org/ws/repo.git") == "ws/repo"
