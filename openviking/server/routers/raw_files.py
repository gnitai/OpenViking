"""Raw file store: writes into the un-indexed `files` scope.

Stores the request body as-is. No parse, no enqueue, no L0/L1, no vectors.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from openviking.core.uri_validation import validate_viking_uri
from openviking.pyagfs.exceptions import AGFSNotFoundError
from openviking.server.dependencies import ensure_project_ready, get_service
from openviking.server.identity import RequestContext
from openviking.server.models import Response
from openviking.storage.transaction import get_lock_manager
from openviking_cli.exceptions import (
    AlreadyExistsError,
    ConflictError,
    InvalidArgumentError,
    InvalidURIError,
)
from openviking_cli.utils.uri import VikingURI

router = APIRouter(prefix="/api/v1/raw-files", tags=["raw-files"])

FILES_ROOT = "wfs://files/"
MAX_BYTES = 20 * 1024 * 1024


def _files_uri(uri: str) -> str:
    """Normalized URI of a path strictly under wfs://files/, else INVALID_URI."""
    uri = VikingURI(validate_viking_uri(uri, allowed_scopes="files")).uri.rstrip("/")
    if not uri.startswith(FILES_ROOT):
        raise InvalidURIError(uri, f"uri must name a path under {FILES_ROOT}")
    return uri


async def _read_body(request: Request) -> bytes:
    """The raw body, refused with 413 past MAX_BYTES before it is all read (uvicorn has no cap)."""
    too_large = HTTPException(status_code=413, detail=f"body is over {MAX_BYTES} bytes")
    if int(request.headers.get("content-length") or 0) > MAX_BYTES:
        raise too_large
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > MAX_BYTES:
            raise too_large
    return bytes(body)


async def _is_taken(viking_fs, uri: str, ctx: RequestContext) -> bool:
    """Only a clean not-found frees a name; any other stat failure raises (VikingFS.exists
    would say False and let the write overwrite)."""
    try:
        await viking_fs.stat(uri, ctx=ctx)
    except AGFSNotFoundError:
        return False
    return True


@router.put("")
async def put_file(
    request: Request,
    uri: str = Query(..., description="Viking URI under wfs://files/"),
    is_dir: bool = Query(False, alias="dir", description="Claim an empty folder (empty body)"),
    ctx: RequestContext = Depends(ensure_project_ready),
):
    uri = _files_uri(uri)
    body = await _read_body(request)
    if is_dir and body:
        raise InvalidArgumentError("dir=true takes an empty body")
    viking_fs = get_service().viking_fs
    # The lock makes a missing parent folder itself but fails when a concurrent upload made it
    # first, so make it here, where "already exists" is fine.
    await viking_fs.mkdir(uri.rsplit("/", 1)[0], exist_ok=True, ctx=ctx)
    # Create-only: exists + write under one exact-path lock, never overwrite.
    lock_manager = get_lock_manager()
    handle = lock_manager.create_handle()
    try:
        if not await lock_manager.acquire_exact_path(handle, viking_fs._uri_to_path(uri, ctx=ctx)):
            raise ConflictError(f"{uri} is being written", resource=uri)
        if await _is_taken(viking_fs, uri, ctx):
            raise AlreadyExistsError(uri, "folder" if is_dir else "file")
        if is_dir:
            await viking_fs.mkdir(uri, ctx=ctx)
        else:
            await viking_fs.write_file_bytes(uri, body, ctx=ctx)
    finally:
        await lock_manager.release(handle)
    result = {"uri": uri} if is_dir else {"uri": uri, "size": len(body)}
    return Response(status="ok", result=result)
