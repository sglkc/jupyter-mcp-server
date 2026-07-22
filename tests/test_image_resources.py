# Copyright (c) 2024- Datalayer, Inc.
#
# BSD 3-Clause License

"""Tests for MCP resource delivery of cell images."""

import base64
import io

import pytest
from PIL import Image

from jupyter_mcp_server.image_resource_store import (
    URI_PREFIX,
    get_image_resource_store,
    publish_cell_image_resource,
)
from jupyter_mcp_server.tools.read_cell_image_tool import ReadCellImageTool
from jupyter_mcp_server.tools._base import ServerMode
from jupyter_mcp_server.models import Notebook, Cell
import jupyter_mcp_server.tools.read_cell_image_tool as tool_mod


def _png_b64(width=40, height=30) -> str:
    img = Image.new("RGB", (width, height), color=(20, 40, 60))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@pytest.fixture(autouse=True)
def _clear_store():
    store = get_image_resource_store()
    store.clear()
    # Also clear FastMCP resource manager entries for our prefix
    try:
        from jupyter_mcp_server.server import mcp

        to_del = [
            u
            for u in list(mcp._resource_manager._resources)
            if str(u).startswith(URI_PREFIX)
        ]
        for u in to_del:
            del mcp._resource_manager._resources[u]
    except Exception:
        pass
    yield
    store.clear()


@pytest.mark.asyncio
async def test_publish_registers_listable_resource():
    raw = base64.b64decode(_png_b64())
    entry = publish_cell_image_resource(
        raw,
        "image/png",
        cell_index=2,
        image_index=0,
        notebook_path="demo.ipynb",
    )
    assert entry.uri.startswith(URI_PREFIX)
    assert get_image_resource_store().get(entry.resource_id) is not None

    from jupyter_mcp_server.server import mcp

    resources = await mcp.list_resources()
    uris = [str(r.uri) for r in resources]
    assert entry.uri in uris

    contents = list(await mcp.read_resource(entry.uri))
    assert len(contents) == 1
    assert contents[0].content == raw
    assert contents[0].mime_type == "image/png"


@pytest.mark.asyncio
async def test_read_cell_image_delivery_resource():
    b64 = _png_b64(80, 50)
    nb = Notebook(
        cells=[
            Cell(
                index=0,
                cell_type="code",
                source="plot()",
                outputs=[
                    {
                        "output_type": "display_data",
                        "data": {"image/png": b64},
                        "metadata": {},
                    }
                ],
            )
        ]
    )

    class FakeCM:
        async def get(self, path, content=True, type="notebook"):
            return {"content": nb.model_dump()}

    orig = tool_mod.get_current_notebook_context
    tool_mod.get_current_notebook_context = lambda notebook_manager=None: (
        "demo.ipynb",
        None,
    )
    try:
        result = await ReadCellImageTool().execute(
            mode=ServerMode.JUPYTER_SERVER,
            contents_manager=FakeCM(),
            notebook_manager=None,
            cell_index=0,
            image_index=0,
            delivery="resource",
        )
    finally:
        tool_mod.get_current_notebook_context = orig

    assert all(isinstance(x, str) for x in result)
    assert any("MCP resource" in x for x in result)
    uri_line = next(x for x in result if x.startswith("uri: "))
    uri = uri_line.split("uri: ", 1)[1].strip()
    assert uri.startswith(URI_PREFIX)

    from jupyter_mcp_server.server import mcp

    contents = list(await mcp.read_resource(uri))
    assert isinstance(contents[0].content, (bytes, bytearray))
    assert len(contents[0].content) > 0
