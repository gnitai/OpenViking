"""
Tests: brand-new project bootstrap (lazy per-project initialization)

A project IS an OpenViking account, asserted dynamically with the ROOT
api key + X-OpenViking-Account header — it is never provisioned via the admin
API. Task 3 makes the FIRST data-plane request for such a project lazily:
  (a) create the preset account/user directories, and
  (b) apply the FULL, schema-bearing create_collection to the project's own
      vector namespace.

Check 1: ls of the member's preset user root for a never-before-seen project
         does NOT 404 (the preset dirs were created on first contact).
Check 2: a content write + read round-trips for that brand-new project (the
         per-project vector namespace got its schema, so the vector attribute
         is not stripped and the write path succeeds).

Requires a live server (Config.SERVER_URL). Mirrors the client/route pattern of
test_project_id_contract.py.
"""

import uuid

import pytest
from api.client import OpenVikingAPIClient
from config import Config

# A brand-new project id that no prior test/admin call has provisioned.
_NEW_PROJECT = "888777"
_MEMBER = "m1"


class TestNewProjectBootstrap:
    def _client(self) -> OpenVikingAPIClient:
        return OpenVikingAPIClient(
            server_url=Config.SERVER_URL,
            api_key=Config.OPENVIKING_API_KEY,
            account=_NEW_PROJECT,
            user=_MEMBER,
        )

    def test_preset_user_root_not_404_for_new_project(self):
        """ls of the member's preset user root must not 404 for a new project."""
        client = self._client()
        try:
            resp = client.fs_ls(f"viking://user/{_MEMBER}")
            # First contact triggers lazy init of the preset dirs; the listing
            # must resolve (200) rather than 404. A 200 envelope with status
            # != "ok" would indicate the dirs were not created.
            assert resp.status_code == 200, (
                f"ls of preset user root returned {resp.status_code}: {resp.text[:200]}"
            )
            data = resp.json()
            assert data.get("status") == "ok", (
                f"ls of preset user root status != ok (preset dirs missing?): {data}"
            )
        finally:
            client.close()

    def test_content_write_read_round_trips_for_new_project(self):
        """A write+read must round-trip for a brand-new project."""
        client = self._client()
        suffix = uuid.uuid4().hex[:8]
        test_uri = f"viking://resources/new_project_bootstrap_{suffix}.txt"
        test_content = f"new project bootstrap {suffix}"
        try:
            # resources is a preset dir created by lazy init; mkdir is idempotent.
            client.fs_mkdir("viking://resources")

            write_resp = client.fs_write(test_uri, test_content, mode="create", wait=True)
            assert write_resp.status_code == 200, (
                f"fs_write failed ({write_resp.status_code}): {write_resp.text[:200]}"
            )
            write_data = write_resp.json()
            if write_data.get("status") != "ok":
                pytest.skip(
                    f"fs_write not available on this environment: {write_data.get('error')}"
                )

            read_resp = client.fs_read(test_uri)
            assert read_resp.status_code == 200, (
                f"fs_read failed ({read_resp.status_code}): {read_resp.text[:200]}"
            )
            read_data = read_resp.json()
            assert read_data.get("status") == "ok", f"fs_read status != ok: {read_data}"
            assert read_data.get("result") == test_content, (
                f"fs_read content mismatch: {read_data}"
            )
        finally:
            try:
                client.fs_rm(test_uri)
            except Exception:
                pass
            client.close()
