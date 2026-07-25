# Copyright (c) 2024- Datalayer, Inc.
#
# BSD 3-Clause License

"""Unit tests for execute_multiple_cells range resolution and stop-on-error."""

from __future__ import annotations

import contextlib
from typing import Any, List
from unittest.mock import AsyncMock, patch

import pytest

from jupyter_mcp_server.tools._base import ServerMode
from jupyter_mcp_server.tools.execute_multiple_cells_tool import (
    ExecuteMultipleCellsTool,
    outputs_indicate_error,
    resolve_cell_range,
)


class FakeNotebook:
    def __init__(self, cells: List[dict]):
        self._cells = cells

    def __len__(self):
        return len(self._cells)

    def __getitem__(self, index):
        return self._cells[index]

    def as_dict(self):
        return {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {},
            "cells": self._cells,
        }


class FakeNotebookManager:
    def __init__(self, notebook: FakeNotebook):
        self._notebook = notebook

    def get_current_notebook(self):
        return "default"

    def get_kernel_id(self, notebook_name):
        return "kernel-1"

    @contextlib.asynccontextmanager
    async def get_current_connection(self):
        yield self._notebook


def test_resolve_range_to_end():
    assert resolve_cell_range(2, None, 5) == (2, 4)


def test_resolve_range_middle():
    assert resolve_cell_range(1, 3, 5) == (1, 3)


def test_resolve_range_prefix():
    assert resolve_cell_range(0, 2, 5) == (0, 2)


def test_resolve_range_rejects_inverted():
    with pytest.raises(ValueError, match="must be >="):
        resolve_cell_range(3, 1, 5)


def test_resolve_range_rejects_oob_start():
    with pytest.raises(ValueError, match="start_index"):
        resolve_cell_range(5, None, 5)


def test_resolve_range_rejects_oob_end():
    with pytest.raises(ValueError, match="end_index"):
        resolve_cell_range(0, 5, 5)


def test_outputs_indicate_error_markers():
    assert outputs_indicate_error(["[ERROR: boom]"]) is True
    assert outputs_indicate_error(["[TIMEOUT ERROR: Cell execution exceeded 1 seconds]"]) is True
    assert outputs_indicate_error(["ok", "42"]) is False
    assert outputs_indicate_error(['print("Error: not a failure")']) is False


@pytest.mark.asyncio
async def test_skips_non_code_and_executes_range():
    cells = [
        {"cell_type": "markdown", "source": "# title", "metadata": {}, "outputs": []},
        {
            "cell_type": "code",
            "source": "1",
            "metadata": {},
            "outputs": [],
            "execution_count": None,
        },
        {
            "cell_type": "code",
            "source": "2",
            "metadata": {},
            "outputs": [],
            "execution_count": None,
        },
        {
            "cell_type": "code",
            "source": "3",
            "metadata": {},
            "outputs": [],
            "execution_count": None,
        },
    ]
    manager = FakeNotebookManager(FakeNotebook(cells))
    executed: list[int] = []

    async def fake_execute(self, **kwargs):
        idx = kwargs["cell_index"]
        executed.append(idx)
        return [f"out-{idx}"]

    with patch(
        "jupyter_mcp_server.tools.execute_multiple_cells_tool.ExecuteCellTool.execute",
        new=fake_execute,
    ):
        # Patch Notebook model construction path by stubbing _get_cell_types
        tool = ExecuteMultipleCellsTool()
        tool._get_cell_types = AsyncMock(  # type: ignore[method-assign]
            return_value=["markdown", "code", "code", "code"]
        )
        tool._notebook_cell_has_error = AsyncMock(return_value=False)  # type: ignore[method-assign]

        result = await tool.execute(
            mode=ServerMode.MCP_SERVER,
            notebook_manager=manager,
            start_index=0,
            end_index=2,
            timeout_seconds=30,
            ensure_kernel_alive_fn=lambda: object(),
        )

    assert executed == [1, 2]
    text = "\n".join(str(x) for x in result)
    assert "skipped" in text
    assert "Completed cells 0..2" in text
    assert "out-1" in text
    assert "out-2" in text


@pytest.mark.asyncio
async def test_stops_on_error_and_skips_remaining():
    tool = ExecuteMultipleCellsTool()
    tool._get_cell_types = AsyncMock(return_value=["code", "code", "code"])  # type: ignore[method-assign]
    tool._notebook_cell_has_error = AsyncMock(return_value=False)  # type: ignore[method-assign]
    executed: list[int] = []

    async def fake_execute(self, **kwargs):
        idx = kwargs["cell_index"]
        executed.append(idx)
        if idx == 1:
            return ["[ERROR: ZeroDivisionError: division by zero]"]
        return [f"out-{idx}"]

    with patch(
        "jupyter_mcp_server.tools.execute_multiple_cells_tool.ExecuteCellTool.execute",
        new=fake_execute,
    ):
        result = await tool.execute(
            mode=ServerMode.MCP_SERVER,
            notebook_manager=FakeNotebookManager(FakeNotebook([])),
            start_index=0,
            end_index=None,
            timeout_seconds=30,
            ensure_kernel_alive_fn=lambda: object(),
        )

    assert executed == [0, 1]
    text = "\n".join(str(x) for x in result)
    assert "Stopped at cell 1 due to error" in text
    assert "out-0" in text
    assert "out-2" not in text


@pytest.mark.asyncio
async def test_stops_when_execute_raises():
    tool = ExecuteMultipleCellsTool()
    tool._get_cell_types = AsyncMock(return_value=["code", "code"])  # type: ignore[method-assign]
    executed: list[int] = []

    async def fake_execute(self, **kwargs):
        idx = kwargs["cell_index"]
        executed.append(idx)
        if idx == 0:
            raise RuntimeError("kernel died")
        return ["should-not-run"]

    with patch(
        "jupyter_mcp_server.tools.execute_multiple_cells_tool.ExecuteCellTool.execute",
        new=fake_execute,
    ):
        result = await tool.execute(
            mode=ServerMode.MCP_SERVER,
            notebook_manager=FakeNotebookManager(FakeNotebook([])),
            start_index=0,
            end_index=1,
            timeout_seconds=30,
            ensure_kernel_alive_fn=lambda: object(),
        )

    assert executed == [0]
    text = "\n".join(str(x) for x in result)
    assert "Stopped at cell 0 due to error" in text
    assert "kernel died" in text


@pytest.mark.asyncio
async def test_start_to_end_omitted_end_index():
    tool = ExecuteMultipleCellsTool()
    tool._get_cell_types = AsyncMock(return_value=["code", "code", "code"])  # type: ignore[method-assign]
    tool._notebook_cell_has_error = AsyncMock(return_value=False)  # type: ignore[method-assign]
    executed: list[int] = []

    async def fake_execute(self, **kwargs):
        executed.append(kwargs["cell_index"])
        return ["ok"]

    with patch(
        "jupyter_mcp_server.tools.execute_multiple_cells_tool.ExecuteCellTool.execute",
        new=fake_execute,
    ):
        result = await tool.execute(
            mode=ServerMode.MCP_SERVER,
            notebook_manager=FakeNotebookManager(FakeNotebook([])),
            start_index=1,
            end_index=None,
            timeout_seconds=10,
            ensure_kernel_alive_fn=lambda: object(),
        )

    assert executed == [1, 2]
    assert any("Completed cells 1..2" in str(x) for x in result)


@pytest.mark.asyncio
async def test_forwards_stream_options_to_execute_cell():
    """stream / progress_interval must be passed through to each ExecuteCellTool call."""
    tool = ExecuteMultipleCellsTool()
    tool._get_cell_types = AsyncMock(return_value=["code", "code"])  # type: ignore[method-assign]
    tool._notebook_cell_has_error = AsyncMock(return_value=False)  # type: ignore[method-assign]
    seen: list[dict[str, Any]] = []

    async def fake_execute(self, **kwargs):
        seen.append(
            {
                "cell_index": kwargs["cell_index"],
                "stream": kwargs.get("stream"),
                "progress_interval": kwargs.get("progress_interval"),
            }
        )
        return [f"out-{kwargs['cell_index']}"]

    with patch(
        "jupyter_mcp_server.tools.execute_multiple_cells_tool.ExecuteCellTool.execute",
        new=fake_execute,
    ):
        await tool.execute(
            mode=ServerMode.MCP_SERVER,
            notebook_manager=FakeNotebookManager(FakeNotebook([])),
            start_index=0,
            end_index=1,
            timeout_seconds=30,
            stream=True,
            progress_interval=3,
            ensure_kernel_alive_fn=lambda: object(),
        )

    assert seen == [
        {"cell_index": 0, "stream": True, "progress_interval": 3},
        {"cell_index": 1, "stream": True, "progress_interval": 3},
    ]


@pytest.mark.asyncio
async def test_stream_defaults_to_false():
    """Default behavior remains non-streaming (matches prior hard-coded False)."""
    tool = ExecuteMultipleCellsTool()
    tool._get_cell_types = AsyncMock(return_value=["code"])  # type: ignore[method-assign]
    tool._notebook_cell_has_error = AsyncMock(return_value=False)  # type: ignore[method-assign]
    seen: list[dict[str, Any]] = []

    async def fake_execute(self, **kwargs):
        seen.append(
            {
                "stream": kwargs.get("stream"),
                "progress_interval": kwargs.get("progress_interval"),
            }
        )
        return ["ok"]

    with patch(
        "jupyter_mcp_server.tools.execute_multiple_cells_tool.ExecuteCellTool.execute",
        new=fake_execute,
    ):
        await tool.execute(
            mode=ServerMode.MCP_SERVER,
            notebook_manager=FakeNotebookManager(FakeNotebook([])),
            start_index=0,
            end_index=0,
            timeout_seconds=10,
            ensure_kernel_alive_fn=lambda: object(),
        )

    assert seen == [{"stream": False, "progress_interval": 5}]
