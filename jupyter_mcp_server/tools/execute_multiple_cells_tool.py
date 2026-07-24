# Copyright (c) 2024- Datalayer, Inc.
#
# BSD 3-Clause License

"""Execute a contiguous range of notebook cells, stopping on the first error."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Union

import nbformat
from jupyter_core.utils import ensure_async
from mcp.types import ImageContent

from jupyter_mcp_server.models import Notebook
from jupyter_mcp_server.tools._base import BaseTool, ServerMode
from jupyter_mcp_server.tools.execute_cell_tool import ExecuteCellTool
from jupyter_mcp_server.utils import get_current_notebook_context

logger = logging.getLogger(__name__)


def outputs_indicate_error(outputs: List[Union[str, ImageContent]]) -> bool:
    """Return True when ExecuteCellTool's extracted outputs signal failure.

    ExecuteCellTool often returns string markers for timeouts/errors instead of
    raising. Only those markers count — plain text that mentions "Error" does not.
    """
    for item in outputs or []:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if text.startswith("[TIMEOUT ERROR:") or text.startswith("[TIMEOUT at"):
            return True
        if text.startswith("[ERROR:"):
            return True
    return False


def resolve_cell_range(
    start_index: int,
    end_index: Optional[int],
    num_cells: int,
) -> tuple[int, int]:
    """Validate and normalize an inclusive [start, end] cell range.

    ``end_index`` of ``None`` means the last cell (index ``num_cells - 1``).
    """
    if num_cells <= 0:
        raise ValueError("Notebook has no cells to execute")
    if start_index < 0:
        raise ValueError(f"start_index must be >= 0 (got {start_index})")
    if start_index >= num_cells:
        raise ValueError(
            f"start_index {start_index} is out of range (notebook has {num_cells} cells)"
        )

    resolved_end = (num_cells - 1) if end_index is None else end_index
    if resolved_end < 0:
        raise ValueError(f"end_index must be >= 0 when set (got {end_index})")
    if resolved_end >= num_cells:
        raise ValueError(
            f"end_index {resolved_end} is out of range (notebook has {num_cells} cells)"
        )
    if resolved_end < start_index:
        raise ValueError(
            f"end_index {resolved_end} must be >= start_index {start_index}"
        )
    return start_index, resolved_end


def _cell_outputs_contain_error(outputs: Any) -> bool:
    """True if notebook cell outputs include an error output_type."""
    if not outputs:
        return False
    for output in outputs:
        if isinstance(output, dict):
            if output.get("output_type") == "error":
                return True
        else:
            # nbformat output objects support attribute access
            if getattr(output, "output_type", None) == "error":
                return True
            try:
                if output.get("output_type") == "error":  # type: ignore[union-attr]
                    return True
            except Exception:
                pass
    return False


class ExecuteMultipleCellsTool(BaseTool):
    """Execute cells in an inclusive index range; stop on the first error."""

    async def _get_cell_types(
        self,
        mode: ServerMode,
        contents_manager: Any,
        notebook_manager: Any,
    ) -> List[str]:
        """Return cell_type for every cell in the active notebook."""
        if mode == ServerMode.JUPYTER_SERVER and contents_manager is not None:
            notebook_path, _ = get_current_notebook_context(notebook_manager)
            if not notebook_path:
                raise ValueError(
                    "No active notebook. Use the use_notebook tool to activate a notebook first."
                )
            model = await ensure_async(
                contents_manager.get(notebook_path, content=True, type="notebook")
            )
            if "content" not in model:
                from jupyter_mcp_server.jupyter_extension.context import get_server_context

                context = get_server_context()
                serverapp = context.serverapp
                path = notebook_path
                if serverapp and not Path(path).is_absolute():
                    path = str(Path(serverapp.root_dir) / path)
                with open(path, "r", encoding="utf-8") as f:
                    nb = nbformat.read(f, as_version=4)
                return [cell.cell_type for cell in nb.cells]
            notebook = Notebook(**model["content"])
            return [notebook[i].cell_type for i in range(len(notebook))]

        if mode == ServerMode.MCP_SERVER and notebook_manager is not None:
            async with notebook_manager.get_current_connection() as notebook_content:
                notebook = Notebook(**notebook_content.as_dict())
            return [notebook[i].cell_type for i in range(len(notebook))]

        raise ValueError(f"Invalid mode or missing required managers: mode={mode}")

    async def _notebook_cell_has_error(
        self,
        mode: ServerMode,
        contents_manager: Any,
        notebook_manager: Any,
        cell_index: int,
    ) -> bool:
        """Inspect the cell after execution for a kernel error output."""
        try:
            if mode == ServerMode.MCP_SERVER and notebook_manager is not None:
                async with notebook_manager.get_current_connection() as notebook:
                    outputs = notebook[cell_index].get("outputs", [])
                    return _cell_outputs_contain_error(outputs)

            if mode == ServerMode.JUPYTER_SERVER and contents_manager is not None:
                notebook_path, _ = get_current_notebook_context(notebook_manager)
                if not notebook_path:
                    return False
                model = await ensure_async(
                    contents_manager.get(notebook_path, content=True, type="notebook")
                )
                if "content" not in model:
                    return False
                notebook = Notebook(**model["content"])
                cell = notebook[cell_index]
                if cell.cell_type != "code":
                    return False
                # Cell model exposes outputs via get_outputs or attribute
                raw = getattr(cell, "outputs", None)
                if raw is None and hasattr(cell, "model"):
                    raw = cell.model.get("outputs", [])
                if raw is None:
                    # Notebook cell dataclass — try as dict-like
                    try:
                        raw = cell["outputs"]  # type: ignore[index]
                    except Exception:
                        raw = []
                return _cell_outputs_contain_error(raw)
        except Exception as exc:
            logger.debug("Could not inspect cell %s for error outputs: %s", cell_index, exc)
        return False

    async def execute(
        self,
        mode: ServerMode,
        server_client=None,
        contents_manager=None,
        kernel_manager=None,
        kernel_spec_manager=None,
        notebook_manager=None,
        serverapp=None,
        start_index: int = 0,
        end_index: Optional[int] = None,
        timeout_seconds: int = 60,
        ensure_kernel_alive_fn=None,
        **kwargs,
    ) -> List[Union[str, ImageContent]]:
        """Execute cells from start_index through end_index (inclusive).

        Non-code cells in the range are skipped. On the first code-cell error
        (kernel exception, timeout, or raised tool failure), execution stops
        and results collected so far are returned.
        """
        cell_types = await self._get_cell_types(mode, contents_manager, notebook_manager)
        start, end = resolve_cell_range(start_index, end_index, len(cell_types))

        execute_cell = ExecuteCellTool()
        result: List[Union[str, ImageContent]] = [
            f"Executing cells {start}..{end} (inclusive) of {len(cell_types)} total; stop on error."
        ]
        executed = 0
        skipped = 0

        for cell_index in range(start, end + 1):
            cell_type = cell_types[cell_index]
            if cell_type != "code":
                skipped += 1
                result.append(f"===== Cell {cell_index} | type: {cell_type} | skipped =====")
                continue

            result.append(f"===== Cell {cell_index} | type: code | executing =====")
            try:
                cell_outputs = await execute_cell.execute(
                    mode=mode,
                    server_client=server_client,
                    contents_manager=contents_manager,
                    kernel_manager=kernel_manager,
                    kernel_spec_manager=kernel_spec_manager,
                    notebook_manager=notebook_manager,
                    serverapp=serverapp,
                    cell_index=cell_index,
                    timeout_seconds=timeout_seconds,
                    stream=False,
                    progress_interval=0,
                    ensure_kernel_alive_fn=ensure_kernel_alive_fn,
                )
            except Exception as exc:
                logger.error("Cell %s raised during multi-execute: %s", cell_index, exc)
                result.append(f"[ERROR: {exc}]")
                result.append(
                    f"Stopped at cell {cell_index} due to error "
                    f"({executed} code cell(s) completed, {skipped} skipped)."
                )
                return result

            if not cell_outputs:
                result.append("[No output generated]")
            else:
                result.extend(cell_outputs)

            has_error = outputs_indicate_error(
                cell_outputs if isinstance(cell_outputs, list) else [cell_outputs]
            )
            if not has_error:
                has_error = await self._notebook_cell_has_error(
                    mode, contents_manager, notebook_manager, cell_index
                )

            if has_error:
                result.append(
                    f"Stopped at cell {cell_index} due to error "
                    f"({executed} code cell(s) completed before this failure, {skipped} skipped)."
                )
                return result

            executed += 1

        result.append(
            f"Completed cells {start}..{end}: {executed} code cell(s) executed, {skipped} non-code skipped."
        )
        return result
