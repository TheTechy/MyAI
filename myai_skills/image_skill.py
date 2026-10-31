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
import os
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

# Where delivered images are written so they survive a page reload (F91 #1).
# Same env var as core/file_generation.py, so images and generated files share
# one output directory — but this stays self-contained (no core.* import).
OUTPUT_DIR = Path(os.getenv("FILE_OUTPUT_DIR", "generated_files"))


def _save_to_disk(filename: str, data: bytes) -> Path:
    """Write *data* to OUTPUT_DIR / filename (creating the dir), avoiding clobber
    by suffixing _1, _2, …. Mirrors file_generation._save_to_disk."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUTPUT_DIR / filename
    stem, suffix = dest.stem, dest.suffix
    counter = 1
    while dest.exists():
        dest = OUTPUT_DIR / f"{stem}_{counter}{suffix}"
        counter += 1
    dest.write_bytes(data)
    return dest


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

        if re.search(r'(\d+)\s*[x×]\s*(\d+)', q) or re.search(r'\b(resize|scale)\b', q):
            return self._apply_resize(img, stem, original_format, q)

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
    # Natural-language scale words → fraction of the original size.
    _SCALE_WORDS = {
        "three quarters": 0.75, "three-quarters": 0.75,
        "quarter": 0.25, "half": 0.5, "double": 2.0, "twice": 2.0,
    }

    def _parse_resize(self, q: str, img: "Image.Image"):
        """Determine target (w, h) from the query, preserving aspect ratio for
        single-dimension / percentage / word forms. Returns (w, h) or None when
        no size can be determined (caller then asks the user)."""
        aspect = (img.height / img.width) if img.width else 1.0

        # 1) Exact WxH — highest priority, unchanged behaviour.
        m = re.search(r'(\d+)\s*[x×]\s*(\d+)', q)
        if m:
            return int(m.group(1)), int(m.group(2))

        # 2) Single dimension with an explicit side → scale the other proportionally.
        m = re.search(r'(?:width|wide)\D{0,6}(\d+)|(\d+)\s*(?:px|pixels?)?\s*(?:wide|width)', q)
        if m:
            w = int(m.group(1) or m.group(2))
            return w, max(1, round(w * aspect))
        m = re.search(r'(?:height|tall|high)\D{0,6}(\d+)|(\d+)\s*(?:px|pixels?)?\s*(?:tall|high|height)', q)
        if m:
            h = int(m.group(1) or m.group(2))
            return max(1, round(h / aspect)), h

        # 3) Percentage → scale both sides.
        m = re.search(r'(\d+(?:\.\d+)?)\s*%', q) or re.search(r'\bby\s+(\d+(?:\.\d+)?)\s*percent\b', q)
        if m:
            frac = float(m.group(1)) / 100.0
            if frac > 0:
                return max(1, round(img.width * frac)), max(1, round(img.height * frac))

        # 4) Natural-language scale words (longest phrases first).
        for word, frac in self._SCALE_WORDS.items():
            if re.search(r'\b' + re.escape(word) + r'\b', q):
                return max(1, round(img.width * frac)), max(1, round(img.height * frac))

        # 5) A lone number ("resize to 400") → ambiguous side; treat as the longest
        #    edge, preserving aspect ratio (predictable, sensible default).
        m = re.search(r'\b(\d{2,})\b', q)
        if m:
            target = int(m.group(1))
            if img.width >= img.height:
                return target, max(1, round(target * aspect))
            return max(1, round(target / aspect)), target

        return None

    def _apply_resize(self, img: Image.Image, stem: str, fmt: str, q: str) -> str:
        target = self._parse_resize(q, img)
        if target is None:
            # No parseable size — ASK rather than silently halving (was the bug).
            return (
                "What size would you like it resized to? You can give exact "
                "dimensions (e.g. '800x600'), a single side (e.g. '400px wide' or "
                "'300 tall'), a percentage (e.g. '50%'), or a word like 'half' or "
                "'quarter'."
            )
        w, h = target
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
        """Encode processed image as base64 and return SKILL_DELIVER string.

        Also writes the image to disk and appends its path as a 5th field so the
        pipeline can persist it (message_files row) and it survives a page reload
        (F91 #1). Payload: SKILL_DELIVER|filename|mime|<base64>|<disk_path>.
        """
        buf      = io.BytesIO()
        save_fmt = "JPEG" if fmt.upper() in ("JPG", "JPEG") else fmt.upper()
        if save_fmt == "JPEG" and img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.save(buf, format=save_fmt)
        data = buf.getvalue()
        b64  = base64.b64encode(data).decode("ascii")
        mime = _MIME.get(save_fmt, "image/png")

        # Persist to disk; on failure, deliver in-session only (empty path) rather
        # than fail the whole request — degrades to the pre-#1 behaviour.
        try:
            disk_path = str(_save_to_disk(filename, data))
        except Exception as e:                                # noqa: BLE001
            print(f"[image_skill] disk save failed ({e}); delivering without persistence")
            disk_path = ""

        return f"SKILL_DELIVER|{filename}|{mime}|{b64}|{disk_path}\n---\n{description}"