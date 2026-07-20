# Copyright (c) 2024- Datalayer, Inc.
#
# BSD 3-Clause License

"""Unit tests for image placeholder extraction and prepare_image_content."""

import base64
import io

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
