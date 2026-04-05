"""Data models for the icon glossary pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from PIL import Image
from pydantic import BaseModel, Field


@dataclass
class IconCandidate:
    """A picture bbox identified as a potential icon during inventory."""

    game_id: str
    doc_name: str
    page_num: int
    bbox_index: int
    bbox: dict[str, Any]
    area: float                          # bbox area in pts²
    image: Image.Image | None = None     # cropped PIL image (None before cropping)
    dhash: str = ""                      # hex DHash string
    clip_embedding: list[float] = field(default_factory=list)
    on_legend_page: bool = False


class GlossaryEntry(BaseModel):
    """A resolved icon with its game-semantic meaning."""

    id: str = Field(description="Unique identifier (e.g. icon_001)")
    name: str = Field(description="Short label (e.g. 'Fire', 'Shield')")
    meaning: str = Field(description="Game-semantic description of what the icon means")
    source: Literal["legend", "icon_reference", "vlm", "web", "manual"] = Field(
        description="How this meaning was determined"
    )
    source_detail: dict[str, Any] = Field(
        default_factory=dict,
        description="Where the meaning was found (doc_name, page_num, bbox_index)",
    )
    dhash: str = Field(default="", description="Hex DHash for perceptual deduplication")
    clip_embedding: list[float] = Field(
        default_factory=list,
        description="CLIP embedding for semantic search",
    )
    occurrences: list[dict[str, Any]] = Field(
        default_factory=list,
        description="All locations where this icon appears [{doc_name, page_num, bbox_index}]",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in the resolved meaning (1.0 = legend match, lower for VLM/web)",
    )


class Glossary(BaseModel):
    """Complete icon glossary for a game."""

    game_id: str
    version: int = 1
    built_at: str = Field(description="ISO timestamp of when the glossary was built")
    entries: list[GlossaryEntry] = Field(default_factory=list)
    unresolved: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Icons that could not be confidently resolved, for user review",
    )
