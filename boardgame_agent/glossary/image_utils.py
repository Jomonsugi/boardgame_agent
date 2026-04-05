"""Image utilities for the glossary pipeline.

Handles cropping icons from PDFs, perceptual hashing, CLIP embedding,
and rendering pages for VLM analysis. Reuses coordinate conversion logic
from the existing pdf_panel.py.
"""

from __future__ import annotations

import io
from typing import Any

import fitz  # PyMuPDF
import imagehash
from PIL import Image

from boardgame_agent.config import DATA_DIR, PAGE_VISION_DPI


def _get_pdf_path(game_id: str, doc_name: str):
    """Find the PDF file for a document (mirrors pdf_panel.get_pdf_path)."""
    from pathlib import Path
    for subdir in ("docs", "pdfs"):
        p = DATA_DIR / "games" / game_id / subdir / f"{doc_name}.pdf"
        if p.exists():
            return p
    return None


def _get_fitz_page(game_id: str, doc_name: str, page_data: dict):
    """Open the PDF and return the correct fitz page + spread metadata."""
    pdf_path = _get_pdf_path(game_id, doc_name)
    if pdf_path is None:
        return None, None, None, None

    doc = fitz.open(str(pdf_path.resolve()))
    pdf_idx = page_data.get("_pdf_page_index", page_data["page_num"] - 1)
    if pdf_idx >= doc.page_count:
        doc.close()
        return None, None, None, None

    fitz_page = doc[pdf_idx]
    spread_half = page_data.get("_spread_half")
    page_height = fitz_page.rect.height
    page_width = fitz_page.rect.width

    x_offset = 0.0
    if spread_half == "right":
        x_offset = page_width / 2

    return doc, fitz_page, page_height, x_offset


def bbox_area(bbox: dict[str, Any]) -> float:
    """Compute the area of a bbox in pts²."""
    w = abs(bbox["x1"] - bbox["x0"])
    h = abs(bbox["y1"] - bbox["y0"])
    return w * h


def crop_bbox_from_pdf(
    game_id: str,
    doc_name: str,
    page_data: dict[str, Any],
    bbox: dict[str, Any],
    padding: float = 3.0,
    dpi: int = 150,
) -> Image.Image | None:
    """Crop a specific bbox region from a PDF page as a PIL Image.

    Uses the same coordinate conversion as pdf_panel.py:
    Docling (bottom-left origin) → PyMuPDF (top-left origin).
    """
    doc, fitz_page, page_height, x_offset = _get_fitz_page(
        game_id, doc_name, page_data
    )
    if doc is None:
        return None

    try:
        x0 = bbox["x0"] + x_offset - padding
        x1 = bbox["x1"] + x_offset + padding
        # Docling: y increases upward. PyMuPDF: y increases downward.
        top_y0 = page_height - bbox["y1"] - padding
        top_y1 = page_height - bbox["y0"] + padding

        rect = fitz.Rect(
            min(x0, x1), min(top_y0, top_y1),
            max(x0, x1), max(top_y0, top_y1),
        )
        # Clamp to page bounds.
        rect = rect & fitz_page.rect

        pix = fitz_page.get_pixmap(dpi=dpi, clip=rect)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    finally:
        doc.close()


def render_page_for_vlm(
    game_id: str,
    doc_name: str,
    page_data: dict[str, Any],
    dpi: int = PAGE_VISION_DPI,
) -> bytes | None:
    """Render a full page as PNG bytes for sending to a VLM API.

    Handles spread-split pages by clipping to the correct half.
    """
    doc, fitz_page, page_height, x_offset = _get_fitz_page(
        game_id, doc_name, page_data
    )
    if doc is None:
        return None

    try:
        spread_half = page_data.get("_spread_half")
        page_width = fitz_page.rect.width

        if spread_half == "left":
            clip = fitz.Rect(0, 0, page_width / 2, page_height)
        elif spread_half == "right":
            clip = fitz.Rect(page_width / 2, 0, page_width, page_height)
        else:
            clip = fitz_page.rect

        pix = fitz_page.get_pixmap(dpi=dpi, clip=clip)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    finally:
        doc.close()


def compute_dhash(img: Image.Image, hash_size: int = 8) -> str:
    """Compute a perceptual difference hash (DHash) as a hex string."""
    return str(imagehash.dhash(img, hash_size=hash_size))


def hamming_distance(hash1: str, hash2: str) -> int:
    """Compute the Hamming distance between two hex hash strings."""
    h1 = imagehash.hex_to_hash(hash1)
    h2 = imagehash.hex_to_hash(hash2)
    return int(h1 - h2)


# ── CLIP embedding (lazy-loaded) ─────────────────────────────────────────────

_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None


def _ensure_clip():
    """Lazy-load the CLIP model on first use."""
    global _clip_model, _clip_preprocess, _clip_tokenizer
    if _clip_model is not None:
        return

    import open_clip
    import torch

    model_name = "ViT-B-32"
    pretrained = "openai"
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    _clip_tokenizer = open_clip.get_tokenizer(model_name)
    _clip_model.eval()


def compute_clip_embedding(img: Image.Image) -> list[float]:
    """Compute a CLIP embedding for an icon image."""
    import torch

    _ensure_clip()
    device = next(_clip_model.parameters()).device

    image_tensor = _clip_preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = _clip_model.encode_image(image_tensor)
        features = features / features.norm(dim=-1, keepdim=True)

    return features[0].cpu().tolist()


def clip_text_embedding(text: str) -> list[float]:
    """Compute a CLIP text embedding for semantic search."""
    import torch

    _ensure_clip()
    device = next(_clip_model.parameters()).device

    tokens = _clip_tokenizer([text]).to(device)
    with torch.no_grad():
        features = _clip_model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)

    return features[0].cpu().tolist()
