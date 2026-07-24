# Copyright (c) 2024- Datalayer, Inc.
#
# BSD 3-Clause License

"""In-memory MCP resource registry for resized cell images.

Used by ``read_cell_image(delivery="resource")``. Concrete BinaryResources are
registered with FastMCP so ``resources/list`` and ``resources/read`` work for
stdio and streamable-http; the Jupyter extension handler also proxies those
methods to FastMCP.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

URI_PREFIX = "resource://jupyter-mcp/cell-images/"


@dataclass(frozen=True)
class StoredImageResource:
    resource_id: str
    uri: str
    data: bytes
    mime_type: str
    name: str
    description: str
    cell_index: int
    image_index: int
    notebook_path: Optional[str]


class ImageResourceStore:
    """Process-local registry of cell image resources."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, StoredImageResource] = {}

    def put(
        self,
        data: bytes,
        mime_type: str,
        *,
        cell_index: int,
        image_index: int,
        notebook_path: Optional[str] = None,
        max_edge: int = 0,
    ) -> StoredImageResource:
        resource_id = uuid.uuid4().hex[:12]
        uri = f"{URI_PREFIX}{resource_id}"
        notebook = notebook_path or "notebook"
        name = f"cell-{cell_index}-img-{image_index}"
        description = (
            f"Resized image from {notebook} cell {cell_index} "
            f"image_index={image_index} ({mime_type}, {len(data)} bytes"
            f"{f', max_edge={max_edge}' if max_edge else ''})"
        )
        entry = StoredImageResource(
            resource_id=resource_id,
            uri=uri,
            data=data,
            mime_type=mime_type,
            name=name,
            description=description,
            cell_index=cell_index,
            image_index=image_index,
            notebook_path=notebook_path,
        )
        with self._lock:
            self._entries[resource_id] = entry
        logger.info("Registered cell image resource %s (%s bytes)", uri, len(data))
        return entry

    def get(self, resource_id: str) -> Optional[StoredImageResource]:
        with self._lock:
            return self._entries.get(resource_id)

    def get_by_uri(self, uri: str) -> Optional[StoredImageResource]:
        if not uri.startswith(URI_PREFIX):
            return None
        return self.get(uri[len(URI_PREFIX) :])

    def list_entries(self) -> list[StoredImageResource]:
        with self._lock:
            return list(self._entries.values())

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


_STORE = ImageResourceStore()


def get_image_resource_store() -> ImageResourceStore:
    return _STORE


def publish_cell_image_resource(
    data: bytes,
    mime_type: str,
    *,
    cell_index: int,
    image_index: int,
    notebook_path: Optional[str] = None,
    max_edge: int = 0,
) -> StoredImageResource:
    """Store image bytes and register a FastMCP BinaryResource for list/read."""
    store = get_image_resource_store()
    entry = store.put(
        data,
        mime_type,
        cell_index=cell_index,
        image_index=image_index,
        notebook_path=notebook_path,
        max_edge=max_edge,
    )
    try:
        from mcp.server.fastmcp.resources.types import BinaryResource
        from jupyter_mcp_server.server import mcp

        # Replace if somehow re-used (unique ids make this rare).
        uri_str = entry.uri
        existing = mcp._resource_manager._resources.get(uri_str)
        if existing is not None:
            del mcp._resource_manager._resources[uri_str]

        mcp.add_resource(
            BinaryResource(
                uri=entry.uri,
                name=entry.name,
                title=entry.name,
                description=entry.description,
                mime_type=entry.mime_type,
                data=entry.data,
            )
        )
    except Exception as exc:
        logger.warning("Failed to register MCP BinaryResource for %s: %s", entry.uri, exc)
    return entry
