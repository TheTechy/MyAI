"""
skills/image_skill.py
=====================
Image manipulation skill using Pillow.

Handles
-------
  Resize       : Resize this image to 800x600
  Convert      : Convert this PNG to JPEG
  Greyscale    : Make this image greyscale / black and white
  Crop         : Crop this image to a square
  Info         : Get the dimensions and file info for this image
  Rotate       : Rotate this image 90 degrees
  Thumbnail    : Create a thumbnail at 200x200

Dependencies
------------
    pip install Pillow
"""

from __future__ import annotations

import base64
import io
import re
from pathlib import Path

from .base import BaseSkill

try:
    from PIL import Image, ImageOps
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# ── MIME types ────────────────────────────────────────────────────────────────
_MIME = {
    "JPEG": "image/jpeg", "JPG": "image/jpeg",
    "PNG":  "image/png",  "GIF": "image/gif",
    "WEBP": "image/webp", "BMP": "image/bmp",
}

_FORMAT_ALIASES = {
    "jpg": "JPEG", "jpeg": "JPEG", "png": "PNG",
    "gif": "GIF",  "webp": "WEBP", "bmp": "BMP",
}


class ImageSkill(BaseSkill):
    name        = "image"
    description = "Manipulate uploaded images — resize, crop, convert, greyscale, rotate"

    def execute(self, query: str, file_contexts: list[dict] | None = None) -> str:
        print(f"[image_skill] execute called")
        if not PIL_AVAILABLE:
            return "Image skill unavailable: Pillow not installed. Run: pip install Pillow"

        if not file_contexts:
            return "No image uploaded. Please attach an image file and try again."


        print(f"[image_skill] execute called, file_contexts={len(file_contexts) if file_contexts else 0}")
        for fc in (file_contexts or []):
            print(f"[image_skill] fc filename={fc.get('filename')} has_image_tag={'[IMAGE]' in fc.get('extracted_text','')}")

        # Find the first image context
        image_ctx = next(
            (fc for fc in file_contexts if "[IMAGE]" in fc.get("extracted_text", "")),
            None
        )
        print(f"[image_skill] image_ctx found: {image_ctx is not None}")
        if not image_ctx:
            return "No image found in the uploaded files. Please attach a .jpg, .png, .webp or similar image."

        # Load image from disk path stored in metadata
        try:
            img, original_format, file_path = self._load_image(image_ctx["extracted_text"])
            print(f"[image_skill] loaded: {file_path} format={original_format} size={img.size}")
        except Exception as e:
            print(f"[image_skill] load error: {e}")
            return f"Could not load image: {e}"

        filename = image_ctx.get("filename", "image.png")
        stem     = filename.rsplit(".", 1)[0]

        # ── Route to handler ──────────────────────────────────────────────────
        q = query.lower()

        if re.search(r'\b(info|information|details|dimensions|size|format|metadata|about)\b', q):
            return self._get_info(img, filename, image_ctx["extracted_text"])

        if re.search(r'\b(grey|gray|greyscale|grayscale|black\s+and\s+white|b&w|bw)\b', q):
            return self._apply_greyscale(img, stem, original_format)

        m = re.search(r'(\d+)\s*[x×]\s*(\d+)', q)
        if m or re.search(r'\b(resize|scale)\b', q):
            return self._apply_resize(img, stem, original_format, m)

        if re.search(r'\b(crop|square)\b', q):
            return self._apply_crop_square(img, stem, original_format)

        m_rot = re.search(r'(\d+)\s*(?:degree|°|deg)', q)
        if m_rot or re.search(r'\b(rotate|rotation)\b', q):
            return self._apply_rotate(img, stem, original_format, m_rot)

        m_fmt = re.search(r'\b(?:to|as|into|convert)\s+(jpe?g|png|webp|gif|bmp)\b', q)
        if m_fmt:
            return self._apply_convert(img, stem, m_fmt.group(1))

        if re.search(r'\b(thumbnail|thumb)\b', q):
            return self._apply_thumbnail(img, stem, original_format)

        return (
            "I can help with: resize (e.g. 'resize to 800x600'), "
            "greyscale, crop to square, rotate (e.g. 'rotate 90 degrees'), "
            "convert (e.g. 'convert to JPEG'), thumbnail, or image info."
        )

    # ── Info ──────────────────────────────────────────────────────────────────
    def _get_info(self, img: Image.Image, filename: str, raw: str) -> str:
        size_line = next((l for l in raw.splitlines() if l.startswith("File size:")), "")
        return (
            f"Image information for {filename}:\n"
            f"  Dimensions : {img.width} × {img.height} px\n"
            f"  Format     : {img.format or 'unknown'}\n"
            f"  Mode       : {img.mode}\n"
            f"  {size_line}"
        )

    # ── Greyscale ─────────────────────────────────────────────────────────────
    def _apply_greyscale(self, img: Image.Image, stem: str, fmt: str) -> str:
        out  = ImageOps.grayscale(img)
        name = f"{stem}_greyscale.{fmt.lower()}"
        return self._deliver(out, name, fmt, f"Converted to greyscale — saved as {name}.")

    # ── Resize ────────────────────────────────────────────────────────────────
    def _apply_resize(self, img: Image.Image, stem: str, fmt: str, m: re.Match | None) -> str:
        if m:
            w, h = int(m.group(1)), int(m.group(2))
        else:
            w, h = img.width // 2, img.height // 2

        out  = img.resize((w, h), Image.LANCZOS)
        name = f"{stem}_{w}x{h}.{fmt.lower()}"
        return self._deliver(out, name, fmt, f"Resized to {w}×{h} px — saved as {name}.")

    # ── Crop to square ────────────────────────────────────────────────────────
    def _apply_crop_square(self, img: Image.Image, stem: str, fmt: str) -> str:
        size = min(img.size)
        out  = ImageOps.fit(img, (size, size))
        name = f"{stem}_square.{fmt.lower()}"
        return self._deliver(out, name, fmt, f"Cropped to {size}×{size} square — saved as {name}.")

    # ── Rotate ────────────────────────────────────────────────────────────────
    def _apply_rotate(self, img: Image.Image, stem: str, fmt: str, m: re.Match | None) -> str:
        degrees = int(m.group(1)) if m else 90
        out     = img.rotate(-degrees, expand=True)
        name    = f"{stem}_rotated{degrees}.{fmt.lower()}"
        return self._deliver(out, name, fmt, f"Rotated {degrees}° — saved as {name}.")

    # ── Convert format ────────────────────────────────────────────────────────
    def _apply_convert(self, img: Image.Image, stem: str, target_ext: str) -> str:
        target_fmt = _FORMAT_ALIASES.get(target_ext.lower(), "JPEG")
        if target_fmt == "JPEG" and img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        name = f"{stem}.{target_ext.lower()}"
        return self._deliver(img, name, target_fmt, f"Converted to {target_fmt} — saved as {name}.")

    # ── Thumbnail ─────────────────────────────────────────────────────────────
    def _apply_thumbnail(self, img: Image.Image, stem: str, fmt: str) -> str:
        out = img.copy()
        out.thumbnail((200, 200), Image.LANCZOS)
        name = f"{stem}_thumb.{fmt.lower()}"
        return self._deliver(out, name, fmt, f"Thumbnail created at {out.width}×{out.height} px — saved as {name}.")

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _load_image(self, extracted_text: str) -> tuple[Image.Image, str, Path]:
        """Parse image path from metadata and load from disk."""
        path_line = next(
            (l for l in extracted_text.splitlines() if l.startswith("Path:")),
            None
        )
        if not path_line:
            raise ValueError("No file path found in image metadata")

        file_path = Path(path_line.replace("Path:", "").strip())
        if not file_path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")

        img = Image.open(file_path)
        img.load()
        fmt = img.format or file_path.suffix.lstrip(".").upper() or "PNG"
        return img, fmt, file_path

    def _deliver(self, img: Image.Image, filename: str, fmt: str, description: str) -> str:
        """Encode processed image as base64 and return SKILL_DELIVER string."""
        buf      = io.BytesIO()
        save_fmt = "JPEG" if fmt.upper() in ("JPG", "JPEG") else fmt.upper()
        if save_fmt == "JPEG" and img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.save(buf, format=save_fmt)
        b64  = base64.b64encode(buf.getvalue()).decode("ascii")
        mime = _MIME.get(save_fmt, "image/png")
        return f"SKILL_DELIVER|{filename}|{mime}|{b64}\n---\n{description}"