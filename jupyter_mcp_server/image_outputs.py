# Copyright (c) 2024- Datalayer, Inc.
#
# BSD 3-Clause License

"""Helpers for notebook image outputs (placeholders, decode, resize)."""

from __future__ import annotations

import base64
import io
import logging
import os
import struct
from typing import Any, Optional, Union

from mcp.types import ImageContent

logger = logging.getLogger(__name__)

# Preferred order when a display_data bundle has multiple image mimes.
IMAGE_MIMES = ("image/png", "image/jpeg", "image/gif")

DEFAULT_MAX_EDGE = int(os.getenv("JUPYTER_MCP_IMAGE_MAX_EDGE", "1024"))
DEFAULT_MAX_BYTES = int(os.getenv("JUPYTER_MCP_IMAGE_MAX_BYTES", "150000"))
DEFAULT_JPEG_QUALITY = int(os.getenv("JUPYTER_MCP_IMAGE_JPEG_QUALITY", "85"))


def _get_env_bool(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    lowered = value.lower().strip()
    if lowered in {"true", "1", "yes", "on", "enable", "enabled"}:
        return True
    if lowered in {"false", "0", "no", "off", "disable", "disabled"}:
        return False
    return default


def images_enabled() -> bool:
    """Whether read_cell_image may return ImageContent (ALLOW_IMG_OUTPUT)."""
    from jupyter_mcp_server.config import ALLOW_IMG_OUTPUT

    return bool(ALLOW_IMG_OUTPUT)


def find_image_in_data(data: Any) -> Optional[tuple[str, str]]:
    """Return (mime, base64_payload) for the first image mime in a data dict."""
    if not isinstance(data, dict):
        return None
    for mime in IMAGE_MIMES:
        if mime in data:
            payload = data[mime]
            if hasattr(payload, "source"):
                payload = str(payload.source)
            if isinstance(payload, list):
                payload = "".join(str(p) for p in payload)
            if payload is None:
                continue
            return mime, str(payload)
    return None


def approx_decoded_bytes(b64_data: str) -> int:
    """Approximate raw byte length of base64 payload."""
    # Strip whitespace that sometimes appears in notebook JSON
    cleaned = "".join(b64_data.split())
    padding = cleaned.count("=")
    return max(0, (len(cleaned) * 3) // 4 - padding)


def _png_dims(raw: bytes) -> Optional[tuple[int, int]]:
    if len(raw) < 24 or raw[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    try:
        width, height = struct.unpack(">II", raw[16:24])
        return int(width), int(height)
    except Exception:
        return None


def _jpeg_dims(raw: bytes) -> Optional[tuple[int, int]]:
    # Minimal SOF0/SOF2 scan
    if len(raw) < 4 or raw[0:2] != b"\xff\xd8":
        return None
    i = 2
    while i + 9 < len(raw):
        if raw[i] != 0xFF:
            i += 1
            continue
        marker = raw[i + 1]
        if marker in (0xC0, 0xC1, 0xC2):  # SOF
            try:
                height, width = struct.unpack(">HH", raw[i + 5 : i + 9])
                return int(width), int(height)
            except Exception:
                return None
        if marker == 0xD9:  # EOI
            break
        if marker == 0x00 or marker == 0x01 or (0xD0 <= marker <= 0xD9):
            i += 2
            continue
        if i + 4 > len(raw):
            break
        seg_len = struct.unpack(">H", raw[i + 2 : i + 4])[0]
        i += 2 + seg_len
    return None


def image_dims(mime: str, b64_data: str) -> Optional[tuple[int, int]]:
    """Best-effort width/height without requiring Pillow."""
    try:
        raw = base64.b64decode(b64_data, validate=False)
    except Exception:
        return None
    if mime == "image/png":
        return _png_dims(raw)
    if mime == "image/jpeg":
        return _jpeg_dims(raw)
    return None


def format_size(num_bytes: int) -> str:
    if num_bytes >= 1024 * 1024:
        return f"~{num_bytes / (1024 * 1024):.1f}MB"
    if num_bytes >= 1024:
        return f"~{num_bytes // 1024}KB"
    return f"~{num_bytes}B"


def format_image_placeholder(
    image_index: int,
    mime: str,
    b64_data: str,
    cell_index: Optional[int] = None,
) -> str:
    """Compact text stand-in for an image so execute/read stay small."""
    nbytes = approx_decoded_bytes(b64_data)
    dims = image_dims(mime, b64_data)
    dim_part = f" {dims[0]}x{dims[1]}" if dims else ""
    size_part = f" {format_size(nbytes)}"
    if cell_index is not None:
        hint = f"read_cell_image(cell_index={cell_index}, image_index={image_index})"
    else:
        hint = f"read_cell_image(image_index={image_index})"
    return f"[image output #{image_index}: {mime}{dim_part}{size_part} — use {hint}]"


def collect_image_outputs(outputs: Any) -> list[tuple[int, str, str, Any]]:
    """Collect (image_index, mime, b64, raw_output) from cell outputs.

    image_index is among image-bearing outputs only (0-based).
    """
    if not outputs:
        return []

    items: list[Any] = []
    if hasattr(outputs, "__iter__") and not isinstance(outputs, (str, dict)):
        try:
            items = list(outputs)
        except Exception:
            items = [outputs]
    else:
        items = [outputs]

    found: list[tuple[int, str, str, Any]] = []
    image_index = 0
    for output in items:
        data = None
        if isinstance(output, dict):
            if output.get("output_type") in ("display_data", "execute_result"):
                data = output.get("data", {})
        else:
            # CRDT / model objects
            try:
                as_dict = output if isinstance(output, dict) else getattr(output, "__dict__", None)
                if hasattr(output, "get"):
                    otype = output.get("output_type") if hasattr(output, "get") else None
                    data = output.get("data") if otype in ("display_data", "execute_result", None) else None
                    if data is None and hasattr(output, "to_py"):
                        py = output.to_py()
                        if isinstance(py, dict) and py.get("output_type") in (
                            "display_data",
                            "execute_result",
                        ):
                            data = py.get("data", {})
                            output = py
            except Exception:
                data = None

        if data is None and isinstance(output, dict):
            data = output.get("data")

        # Normalize CRDT map-like data
        if data is not None and not isinstance(data, dict) and hasattr(data, "to_py"):
            try:
                data = data.to_py()
            except Exception:
                pass

        hit = find_image_in_data(data) if data is not None else None
        if hit:
            mime, b64 = hit
            found.append((image_index, mime, b64, output))
            image_index += 1

    return found


def prepare_image_content(
    mime: str,
    b64_data: str,
    max_edge: int = DEFAULT_MAX_EDGE,
    max_bytes: int = DEFAULT_MAX_BYTES,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> ImageContent:
    """Decode, optionally resize/recompress, return MCP ImageContent.

    Uses Pillow when available. Without Pillow, returns the original payload
    if under max_bytes, otherwise raises ValueError.
    """
    cleaned = "".join(b64_data.split())
    try:
        raw = base64.b64decode(cleaned, validate=False)
    except Exception as exc:
        raise ValueError(f"Invalid base64 image data: {exc}") from exc

    try:
        from PIL import Image
    except ImportError:
        if len(raw) > max_bytes:
            raise ValueError(
                f"Image is {format_size(len(raw))} which exceeds max "
                f"{format_size(max_bytes)} and Pillow is not installed to resize. "
                "Install pillow or raise JUPYTER_MCP_IMAGE_MAX_BYTES."
            )
        return ImageContent(type="image", data=cleaned, mimeType=mime)

    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception as exc:
        raise ValueError(f"Could not decode image: {exc}") from exc

    # Normalize mode for JPEG
    out_mime = mime
    if img.mode not in ("RGB", "L") and mime == "image/jpeg":
        img = img.convert("RGB")

    width, height = img.size
    longest = max(width, height)
    if max_edge > 0 and longest > max_edge:
        scale = max_edge / float(longest)
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    def encode(target_mime: str, quality: int) -> tuple[bytes, str]:
        buf = io.BytesIO()
        if target_mime == "image/jpeg":
            to_save = img if img.mode in ("RGB", "L") else img.convert("RGB")
            to_save.save(buf, format="JPEG", quality=quality, optimize=True)
            return buf.getvalue(), "image/jpeg"
        if target_mime == "image/gif":
            img.save(buf, format="GIF", optimize=True)
            return buf.getvalue(), "image/gif"
        # default PNG
        to_save = img
        if img.mode == "P":
            to_save = img.convert("RGBA")
        to_save.save(buf, format="PNG", optimize=True)
        return buf.getvalue(), "image/png"

    # Prefer keeping original mime when small enough after resize.
    encoded, out_mime = encode(mime if mime in IMAGE_MIMES else "image/png", jpeg_quality)

    # If still too large, try JPEG re-encode at decreasing quality.
    if len(encoded) > max_bytes:
        for quality in (jpeg_quality, 70, 55, 40):
            encoded, out_mime = encode("image/jpeg", quality)
            if len(encoded) <= max_bytes:
                break

    if len(encoded) > max_bytes:
        raise ValueError(
            f"Image still {format_size(len(encoded))} after compression "
            f"(limit {format_size(max_bytes)}). "
            "Lower resolution of the plot or raise JUPYTER_MCP_IMAGE_MAX_BYTES."
        )

    return ImageContent(
        type="image",
        data=base64.b64encode(encoded).decode("ascii"),
        mimeType=out_mime,
    )


def extract_image_from_cell_outputs(
    outputs: Any,
    image_index: int = 0,
) -> tuple[str, str]:
    """Return (mime, b64) for image_index among image outputs; raises ValueError."""
    images = collect_image_outputs(outputs)
    if not images:
        raise ValueError("No image outputs found on this cell.")
    if image_index < 0 or image_index >= len(images):
        raise ValueError(
            f"image_index {image_index} out of range. "
            f"Cell has {len(images)} image output(s) (0..{len(images) - 1})."
        )
    _, mime, b64, _ = images[image_index]
    return mime, b64
