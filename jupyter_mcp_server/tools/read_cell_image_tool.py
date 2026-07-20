# Copyright (c) 2024- Datalayer, Inc.
#
# BSD 3-Clause License

"""Read a single image output from a notebook cell (opt-in vision)."""

from typing import Any, List, Optional, Union

from jupyter_core.utils import ensure_async
from jupyter_server_client import JupyterServerClient
from mcp.types import ImageContent

from jupyter_mcp_server.image_outputs import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_EDGE,
    DeliveryMode,
    collect_image_outputs,
    get_artifact_root,
    images_enabled,
    prepare_image_bytes,
    prepare_image_content,
    write_image_artifact,
)
from jupyter_mcp_server.image_resource_store import publish_cell_image_resource
from jupyter_mcp_server.models import Notebook
from jupyter_mcp_server.notebook_manager import NotebookManager
from jupyter_mcp_server.tools._base import BaseTool, ServerMode
from jupyter_mcp_server.utils import get_current_notebook_context


class ReadCellImageTool(BaseTool):
    """Load one image output from a code cell, resized for agent hosts."""

    async def _load_notebook(
        self,
        mode: ServerMode,
        contents_manager: Optional[Any],
        notebook_manager: Optional[NotebookManager],
    ) -> tuple[Union[Notebook, List[str]], Optional[str]]:
        """Return (notebook_or_error, notebook_path_for_artifacts)."""
        if mode == ServerMode.JUPYTER_SERVER and contents_manager is not None:
            notebook_path, _ = get_current_notebook_context(notebook_manager)
            if not notebook_path:
                return (
                    [
                        "No active notebook. Use the use_notebook tool to activate a notebook first."
                    ],
                    None,
                )
            model = await ensure_async(
                contents_manager.get(notebook_path, content=True, type="notebook")
            )
            if "content" not in model:
                raise ValueError(f"Could not read notebook content from {notebook_path}")
            return Notebook(**model["content"]), notebook_path

        if mode == ServerMode.MCP_SERVER and notebook_manager is not None:
            notebook_path = None
            if hasattr(notebook_manager, "get_current_notebook_path"):
                notebook_path = notebook_manager.get_current_notebook_path()
            async with notebook_manager.get_current_connection() as notebook_content:
                return Notebook(**notebook_content.as_dict()), notebook_path

        raise ValueError(f"Invalid mode or missing required clients: mode={mode}")

    async def execute(
        self,
        mode: ServerMode,
        server_client: Optional[JupyterServerClient] = None,
        kernel_client: Optional[Any] = None,
        contents_manager: Optional[Any] = None,
        kernel_manager: Optional[Any] = None,
        kernel_spec_manager: Optional[Any] = None,
        notebook_manager: Optional[NotebookManager] = None,
        cell_index: int = 0,
        image_index: int = 0,
        max_edge: int = DEFAULT_MAX_EDGE,
        max_bytes: int = DEFAULT_MAX_BYTES,
        delivery: DeliveryMode = "image",
        **kwargs,
    ) -> list[Union[str, ImageContent]]:
        """Return one resized image from a cell's outputs.

        Args:
            cell_index: Notebook cell index (0-based).
            image_index: Index among *image* outputs only (0-based).
            max_edge: Longest side in pixels after resize (0 = no resize).
            max_bytes: Soft max raw bytes after compression.
            delivery: ``image`` returns MCP ImageContent; ``path`` writes under
                ``JUPYTER_MCP_ARTIFACT_DIR`` and returns the absolute path only;
                ``resource`` registers an MCP Resource and returns its URI.
        """
        if not images_enabled():
            return [
                "Image output is disabled (ALLOW_IMG_OUTPUT=false). "
                "Enable it to use read_cell_image."
            ]

        if delivery not in ("image", "path", "resource"):
            return [
                f"Unknown delivery mode {delivery!r}. "
                "Use 'image', 'path', or 'resource'."
            ]

        if delivery == "path" and get_artifact_root() is None:
            return [
                "Path delivery is unavailable: set JUPYTER_MCP_ARTIFACT_DIR to a "
                "writable directory the agent can read (typically when MCP is "
                "stdio-colocated with the agent). Use delivery='image' or "
                "delivery='resource' instead."
            ]

        notebook_or_err, notebook_path = await self._load_notebook(
            mode, contents_manager, notebook_manager
        )
        if isinstance(notebook_or_err, list):
            return notebook_or_err
        notebook = notebook_or_err

        if cell_index < 0 or cell_index >= len(notebook):
            return [
                f"Cell index {cell_index} is out of range. "
                f"Notebook has {len(notebook)} cells."
            ]

        cell = notebook.cells[cell_index]
        if cell.cell_type != "code":
            return [
                f"Cell {cell_index} is a {cell.cell_type} cell, not a code cell. "
                "Only code cells have image outputs."
            ]

        images = collect_image_outputs(cell.outputs)
        if not images:
            return [
                f"Cell {cell_index} has no image outputs. "
                "Execute a cell that produces plots or images first."
            ]
        if image_index < 0 or image_index >= len(images):
            return [
                f"image_index {image_index} out of range for cell {cell_index}. "
                f"This cell has {len(images)} image output(s) "
                f"(valid: 0..{len(images) - 1})."
            ]

        _, mime, b64, _ = images[image_index]

        if delivery in ("path", "resource"):
            try:
                encoded, out_mime = prepare_image_bytes(
                    mime,
                    b64,
                    max_edge=max_edge,
                    max_bytes=max_bytes,
                )
            except ValueError as exc:
                return [f"[ERROR: {exc}]"]

            if delivery == "path":
                try:
                    path = write_image_artifact(
                        encoded,
                        out_mime,
                        cell_index=cell_index,
                        image_index=image_index,
                        notebook_path=notebook_path,
                    )
                except ValueError as exc:
                    return [f"[ERROR: {exc}]"]
                return [
                    f"Cell {cell_index} image #{image_index} saved "
                    f"({out_mime}, max_edge={max_edge}, {len(encoded)} bytes)",
                    f"path: {path}",
                ]

            entry = publish_cell_image_resource(
                encoded,
                out_mime,
                cell_index=cell_index,
                image_index=image_index,
                notebook_path=notebook_path,
                max_edge=max_edge,
            )
            return [
                f"Cell {cell_index} image #{image_index} published as MCP resource "
                f"({out_mime}, max_edge={max_edge}, {len(encoded)} bytes). "
                "Use resources/list or resources/read to fetch the blob.",
                f"uri: {entry.uri}",
            ]

        try:
            content = prepare_image_content(
                mime,
                b64,
                max_edge=max_edge,
                max_bytes=max_bytes,
            )
        except ValueError as exc:
            return [f"[ERROR: {exc}]"]

        meta = (
            f"Cell {cell_index} image #{image_index} "
            f"({content.mimeType}, max_edge={max_edge})"
        )
        return [meta, content]
