"""
Tests: project_id == account_id contract

A project IS an OpenViking account (the tenant boundary).
The caller passes project_id in X-OpenViking-Account with the ROOT api key.

Check 1: Numeric project_id values are accepted by UserIdentifier validation.
Check 2: Absent X-OpenViking-Account header resolves to account_id == "default".
Check 3: Data written under project "3493" is NOT visible under project "999"
         (per-project filesystem isolation).
"""

import uuid

import pytest
import requests
from api.client import OpenVikingAPIClient
from config import Config

from openviking_cli.session.user_id import UserIdentifier


# ---------------------------------------------------------------------------
# Check 1 — pure unit assertion, no server needed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("project_id", ["3493", "923922", "default"])
def test_valid_project_id_accepted_by_user_identifier(project_id):
    """UserIdentifier accepts numeric project_ids as account_id."""
    # The real assertion is that the UserIdentifier constructor does not raise
    # ValueError for these ids (and round-trips them as account_id).
    uid = UserIdentifier(project_id, "member1", "default")
    assert uid.account_id == project_id


# ---------------------------------------------------------------------------
# Check 2 — absent X-OpenViking-Account resolves to "default"
#
# We use GET /health which:
#   a) calls resolve_identity directly and does NOT go through
#      get_request_context, so the explicit-tenant guard never fires; with a
#      ROOT key and no X-OpenViking-Account header, auth.py resolves
#      account_id to "default".
#   b) echoes account_id in the response when an auth header is present
# ---------------------------------------------------------------------------


class TestAbsentAccountHeaderDefaultsToDefault:
    def test_absent_account_header_resolves_to_default(self):
        """ROOT key + no X-OpenViking-Account → account_id == 'default'."""
        # Build a minimal requests session: only Authorization, no account header.
        session = requests.Session()
        session.headers.update(
            {
                "Authorization": f"Bearer {Config.OPENVIKING_API_KEY}",
                "Accept": "*/*",
            }
        )
        url = f"{Config.SERVER_URL}/health"
        resp = session.get(url, timeout=10)
        session.close()

        assert resp.status_code == 200, (
            f"GET /health returned {resp.status_code}: {resp.text[:200]}"
        )
        data = resp.json()
        assert data.get("healthy") is True, f"Expected healthy=true, got: {data}"
        account_id = data.get("account_id")
        assert account_id == "default", (
            f"Expected account_id='default' when X-OpenViking-Account is absent, "
            f"got '{account_id}'. Full response: {data}"
        )


# ---------------------------------------------------------------------------
# Check 3 — data written under project "3493" is NOT visible under "999"
#
# Uses the fs_write / fs_stat pattern from api/client.py.
# Two separate OpenVikingAPIClient instances are constructed — one per project —
# rather than mutating the shared session-scoped fixture.
# ---------------------------------------------------------------------------


class TestPerProjectFilesystemIsolation:
    def test_write_project_3493_not_visible_under_project_999(self):
        """File written as project '3493' must not be readable as project '999'."""
        unique_suffix = uuid.uuid4().hex[:8]
        test_uri = f"wfs://resources/project_id_isolation_test_{unique_suffix}.txt"
        test_content = f"project_id contract test {unique_suffix}"

        client_3493 = OpenVikingAPIClient(
            server_url=Config.SERVER_URL,
            api_key=Config.OPENVIKING_API_KEY,
            account="3493",
            user="m1",
        )
        client_999 = OpenVikingAPIClient(
            server_url=Config.SERVER_URL,
            api_key=Config.OPENVIKING_API_KEY,
            account="999",
            user="m1",
        )

        try:
            # Ensure resources dir exists in project 3493
            client_3493.fs_mkdir("wfs://resources")

            # Write the file as project 3493 (mode="create" for new files)
            write_resp = client_3493.fs_write(test_uri, test_content, mode="create", wait=True)
            assert write_resp.status_code == 200, (
                f"fs_write failed ({write_resp.status_code}): {write_resp.text[:200]}"
            )
            write_data = write_resp.json()
            if write_data.get("status") != "ok":
                # When fs_write is unavailable (no AGFS), the per-project
                # isolation contract is NOT validated in this environment.
                pytest.skip(
                    f"fs_write not available on this environment: {write_data.get('error')}"
                )

            # Attempt to read back under the SAME project — must succeed
            read_same_resp = client_3493.fs_read(test_uri)
            assert read_same_resp.status_code == 200, (
                f"fs_read (same project) failed ({read_same_resp.status_code}): "
                f"{read_same_resp.text[:200]}"
            )
            same_data = read_same_resp.json()
            assert same_data.get("status") == "ok", (
                f"fs_read (same project) status != ok: {same_data}"
            )
            assert same_data.get("result") == test_content, (
                f"fs_read (same project) content mismatch: {same_data}"
            )

            # Attempt to read the same URI under project 999 — must NOT find it
            read_other_resp = client_999.fs_read(test_uri)
            # Either 404 HTTP status OR a 200 envelope with status != "ok" / no result.
            # A 200 with status=="ok" AND result==test_content would be an isolation failure.
            if read_other_resp.status_code == 200:
                other_data = read_other_resp.json()
                is_found = (
                    other_data.get("status") == "ok"
                    and other_data.get("result") == test_content
                )
                assert not is_found, (
                    "ISOLATION FAILURE: data written as project '3493' is visible "
                    f"under project '999'. Response: {other_data}"
                )
            else:
                # 403 allows for a future server enforcing cross-project
                # isolation via an access-denied response.
                assert read_other_resp.status_code in (400, 403, 404), (
                    f"Expected 400/403/404 (not-found or forbidden) when reading "
                    f"cross-project, got {read_other_resp.status_code}: "
                    f"{read_other_resp.text[:200]}"
                )
        finally:
            # Clean up only in the owning project
            try:
                client_3493.fs_rm(test_uri)
            except Exception:
                pass
            client_3493.close()
            client_999.close()
