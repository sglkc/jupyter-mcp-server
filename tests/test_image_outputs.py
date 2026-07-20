# Copyright (c) 2024- Datalayer, Inc.
#
# BSD 3-Clause License

"""Unit tests for image placeholder extraction and prepare_image_content."""

import base64
import io
from pathlib import Path

import pytest
from mcp.types import ImageContent

from jupyter_mcp_server.image_outputs import (
    collect_image_outputs,
    format_image_placeholder,
    prepare_image_content,
)
from jupyter_mcp_server.utils import extract_output, safe_extract_outputs


def _tiny_png_b64(width=32, height=24):
    from PIL import Image

    img = Image.new("RGB", (width, height), color=(10, 20, 30))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_extract_output_image_is_placeholder_not_image_content():
    b64 = _tiny_png_b64()
    output = {
        "output_type": "display_data",
        "data": {"image/png": b64},
        "metadata": {},
    }
    extracted = extract_output(output, image_index=0)
    assert isinstance(extracted, str)
    assert "image output #0" in extracted
    assert "image/png" in extracted
    assert "read_cell_image" in extracted
    assert not isinstance(extracted, ImageContent)


def test_safe_extract_outputs_numbers_multiple_images():
    b64 = _tiny_png_b64()
    outputs = [
        {"output_type": "stream", "name": "stdout", "text": "hello\n"},
        {
            "output_type": "display_data",
            "data": {"image/png": b64},
            "metadata": {},
        },
        {
            "output_type": "display_data",
            "data": {"image/png": b64},
            "metadata": {},
        },
    ]
    result = safe_extract_outputs(outputs)
    assert result[0] == "hello\n" or result[0].strip() == "hello"
    images = [r for r in result if isinstance(r, str) and "image output #" in r]
    assert len(images) == 2
    assert "image output #0" in images[0]
    assert "image output #1" in images[1]


def test_extract_output_inline_images_legacy():
    b64 = _tiny_png_b64()
    output = {
        "output_type": "display_data",
        "data": {"image/png": b64},
        "metadata": {},
    }
    extracted = extract_output(output, inline_images=True)
    assert isinstance(extracted, ImageContent)
    assert extracted.mimeType == "image/png"
    assert extracted.data == b64


def test_collect_image_outputs_jpeg_and_png():
    png = _tiny_png_b64()
    # minimal jpeg via pillow
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color="red").save(buf, format="JPEG")
    jpeg = base64.b64encode(buf.getvalue()).decode("ascii")

    outputs = [
        {"output_type": "display_data", "data": {"image/png": png}, "metadata": {}},
        {"output_type": "display_data", "data": {"image/jpeg": jpeg}, "metadata": {}},
    ]
    found = collect_image_outputs(outputs)
    assert len(found) == 2
    assert found[0][1] == "image/png"
    assert found[1][1] == "image/jpeg"


def test_prepare_image_content_resizes():
    b64 = _tiny_png_b64(width=400, height=300)
    content = prepare_image_content("image/png", b64, max_edge=100, max_bytes=150000)
    assert isinstance(content, ImageContent)
    raw = base64.b64decode(content.data)
    from PIL import Image

    img = Image.open(io.BytesIO(raw))
    assert max(img.size) <= 100


def test_format_image_placeholder_includes_hint():
    b64 = _tiny_png_b64()
    text = format_image_placeholder(2, "image/png", b64, cell_index=5)
    assert "image output #2" in text
    assert "cell_index=5" in text
    assert "image_index=2" in text


def test_prepare_image_bytes_and_write_artifact(tmp_path, monkeypatch):
    from jupyter_mcp_server.image_outputs import (
        prepare_image_bytes,
        write_image_artifact,
        get_artifact_root,
    )

    monkeypatch.setenv("JUPYTER_MCP_ARTIFACT_DIR", str(tmp_path))
    # reload constants that read env at call time for get_artifact_root
    assert get_artifact_root() == tmp_path.resolve()

    b64 = _tiny_png_b64(width=200, height=100)
    encoded, mime = prepare_image_bytes("image/png", b64, max_edge=50, max_bytes=150000)
    assert mime.startswith("image/")
    assert len(encoded) > 0

    path = write_image_artifact(
        encoded,
        mime,
        cell_index=3,
        image_index=1,
        notebook_path="analysis/plot demo.ipynb",
    )
    assert path.is_file()
    assert path.parent.name == "plot_demo" or "plot" in path.parent.name
    assert path.name.startswith("cell-3-img-1.")
    assert path.read_bytes() == encoded


def test_write_image_artifact_requires_root(monkeypatch):
    from jupyter_mcp_server.image_outputs import write_image_artifact

    monkeypatch.delenv("JUPYTER_MCP_ARTIFACT_DIR", raising=False)
    with pytest.raises(ValueError, match="JUPYTER_MCP_ARTIFACT_DIR"):
        write_image_artifact(b"x", "image/png", cell_index=0, image_index=0)


@pytest.mark.asyncio
async def test_read_cell_image_delivery_path(tmp_path, monkeypatch):
    from jupyter_mcp_server.tools.read_cell_image_tool import ReadCellImageTool
    from jupyter_mcp_server.tools._base import ServerMode
    from jupyter_mcp_server.models import Notebook, Cell
    import jupyter_mcp_server.tools.read_cell_image_tool as tool_mod

    monkeypatch.setenv("JUPYTER_MCP_ARTIFACT_DIR", str(tmp_path))
    b64 = _tiny_png_b64(width=80, height=40)
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
            max_edge=64,
            max_bytes=150000,
            delivery="path",
        )
    finally:
        tool_mod.get_current_notebook_context = orig

    assert isinstance(result[0], str)
    assert "path:" in result[1]
    saved = result[1].split("path:", 1)[1].strip()
    assert Path(saved).is_file()
    # No ImageContent when delivery=path
    assert all(isinstance(x, str) for x in result)


@pytest.mark.asyncio
async def test_read_cell_image_path_unavailable_without_env(monkeypatch):
    from jupyter_mcp_server.tools.read_cell_image_tool import ReadCellImageTool
    from jupyter_mcp_server.tools._base import ServerMode

    monkeypatch.delenv("JUPYTER_MCP_ARTIFACT_DIR", raising=False)
    result = await ReadCellImageTool().execute(
        mode=ServerMode.JUPYTER_SERVER,
        contents_manager=object(),
        cell_index=0,
        delivery="path",
    )
    assert any("JUPYTER_MCP_ARTIFACT_DIR" in r for r in result if isinstance(r, str))
