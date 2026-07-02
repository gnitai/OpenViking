# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Tests for context_type classification in Summarizer.

Verifies that the summarizer uses context_type_for_uri() to correctly
classify memory, skill, and resource URIs (fixes #1060).
"""

from openviking.core.namespace import context_type_for_uri


class TestSummarizerContextType:
    """Verify URI → context_type mapping used by Summarizer."""

    def test_user_memory_uri(self):
        """User memory URIs should classify as 'memory'."""
        uri = "wfs://user/default/memories/entities/mem_abc123.md"
        assert context_type_for_uri(uri) == "memory"

    def test_agent_memory_uri(self):
        """Agent memory URIs should classify as 'memory'."""
        uri = "wfs://agent/default/memories/cases/mem_xyz789.md"
        assert context_type_for_uri(uri) == "memory"

    def test_memory_profile_uri(self):
        """Profile memory URI should classify as 'memory'."""
        uri = "wfs://user/default/memories/profile.md"
        assert context_type_for_uri(uri) == "memory"

    def test_memory_preferences_uri(self):
        """Preferences memory URI should classify as 'memory'."""
        uri = "wfs://user/default/memories/preferences/coding-style.md"
        assert context_type_for_uri(uri) == "memory"

    def test_skill_uri(self):
        """Skill URIs should classify as 'skill'."""
        uri = "wfs://agent/default/skills/search.md"
        assert context_type_for_uri(uri) == "skill"

    def test_resource_uri(self):
        """Resource URIs should classify as 'resource'."""
        uri = "wfs://resources/docs/readme.md"
        assert context_type_for_uri(uri) == "resource"

    def test_session_uri_defaults_to_resource(self):
        """Session URIs should not be treated as memory-specific content."""
        uri = "wfs://session/default/sess_001/history/archive_001"
        assert context_type_for_uri(uri) == "resource"

    def test_resource_named_memories_does_not_classify_as_memory(self):
        """String containment alone should not classify a resource as memory."""
        uri = "wfs://resources/memories-report.md"
        assert context_type_for_uri(uri) == "resource"

    def test_unknown_uri_defaults_to_resource(self):
        """Unknown URIs should default to 'resource'."""
        uri = "wfs://unknown/something"
        assert context_type_for_uri(uri) == "resource"

    def test_old_broken_prefix_would_fail(self):
        """Regression: the old startswith('wfs://memory/') check would miss real URIs."""
        # These real memory URIs do NOT start with "wfs://memory/"
        real_uris = [
            "wfs://user/default/memories/entities/mem_001.md",
            "wfs://agent/default/memories/patterns/mem_002.md",
            "wfs://user/john/memories/events/mem_003.md",
        ]
        for uri in real_uris:
            assert not uri.startswith("wfs://memory/"), (
                f"URI unexpectedly matches old prefix: {uri}"
            )
            assert context_type_for_uri(uri) == "memory", f"URI should classify as memory: {uri}"
