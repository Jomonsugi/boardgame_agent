"""Agentic glossary builder — extracts icon/symbol glossaries from game documents.

Architecture: a multi-stage pipeline with an agentic resolve loop for
unmatched icons. The pipeline:

  1. analyze   — scan pages, detect legend/reference pages, catalog icons
  2. inventory — crop icons, compute DHash + CLIP embeddings, deduplicate
  3. resolve   — link icons to meanings (legend → hash match → VLM → web)
  4. review    — cross-check consistency, flag unresolved
  5. save      — write glossary.json, optionally trigger reindex
"""

from __future__ import annotations

import json
import math
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from boardgame_agent.config import (
    DATA_DIR,
    GLOSSARY_VLM_MODEL,
    ICON_AREA_MAX,
    ICON_AREA_MIN,
    ICON_HASH_FUZZY_THRESHOLD,
    ICON_HASH_MATCH_THRESHOLD,
    LEGEND_SCORE_THRESHOLD,
)
from boardgame_agent.glossary.image_utils import (
    bbox_area,
    compute_clip_embedding,
    compute_dhash,
    crop_bbox_from_pdf,
    hamming_distance,
    render_page_for_vlm,
)
from boardgame_agent.glossary.models import GlossaryEntry, Glossary, IconCandidate
from boardgame_agent.rag.extractor import load_cached_pages


# ── Stage 1: Analyze Documents ───────────────────────────────────────────────


def _detect_legend_pages(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Score pages for legend/glossary likelihood and return those above threshold."""
    scored: list[tuple[float, dict]] = []

    for page in pages:
        bboxes = page.get("bboxes", [])
        if not bboxes:
            continue

        pictures = [b for b in bboxes if b.get("label") == "picture"]
        texts = [b for b in bboxes if b.get("label") in ("text", "section_header", "caption")]

        small_pictures = [p for p in pictures if ICON_AREA_MIN <= bbox_area(p) <= ICON_AREA_MAX]

        # Heuristic 1: small-picture density
        picture_ratio = len(small_pictures) / max(len(bboxes), 1)

        # Heuristic 2: text brevity (legend pages have short labels, not paragraphs)
        avg_text_len = 0.0
        if texts:
            avg_text_len = sum(len(t.get("text", "")) for t in texts) / len(texts)
        brevity_score = 1.0 if avg_text_len < 50 else (50.0 / max(avg_text_len, 1))

        # Heuristic 3: keywords
        page_text = page.get("text", "").lower()
        keywords = ["legend", "glossary", "icon", "symbol", "overview", "reference"]
        keyword_score = min(sum(1 for kw in keywords if kw in page_text), 3) / 3.0

        # Heuristic 4: grid alignment (low x-position variance among icons)
        grid_score = 0.0
        if len(small_pictures) >= 4:
            x_positions = [(b["x0"] + b["x1"]) / 2 for b in small_pictures]
            mean_x = sum(x_positions) / len(x_positions)
            variance = sum((x - mean_x) ** 2 for x in x_positions) / len(x_positions)
            std_dev = math.sqrt(variance)
            grid_score = max(0, 1.0 - std_dev / 100.0)

        score = (
            picture_ratio * 0.4
            + brevity_score * 0.2
            + keyword_score * 0.1
            + grid_score * 0.3
        )

        if score >= LEGEND_SCORE_THRESHOLD:
            page["_legend_score"] = score
            scored.append((score, page))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [page for _, page in scored]


def _find_icon_reference_docs(
    game_id: str,
) -> list[str]:
    """Find documents tagged as icon_reference or with icon-related names."""
    from boardgame_agent.db.games import get_documents
    from boardgame_agent.config import GAMES_DB_PATH

    docs = get_documents(game_id, GAMES_DB_PATH)
    ref_docs = []
    for d in docs:
        tag = d.get("doc_tag", "")
        name = d.get("doc_name", "").lower()
        if tag == "icon_reference":
            ref_docs.append(d["doc_name"])
        elif any(kw in name for kw in ("icon", "symbol", "reference", "glossary")):
            ref_docs.append(d["doc_name"])
    return ref_docs


# ── Stage 2: Build Icon Inventory ────────────────────────────────────────────


def _inventory_icons(
    game_id: str,
    all_pages: dict[str, list[dict[str, Any]]],
    legend_pages: list[dict[str, Any]],
    on_progress: Any = None,
) -> list[IconCandidate]:
    """Scan all pages for icon-sized picture bboxes, crop, and hash them."""
    legend_keys = {(p["doc_name"], p["page_num"]) for p in legend_pages}
    candidates: list[IconCandidate] = []

    for doc_name, pages in all_pages.items():
        for page in pages:
            bboxes = page.get("bboxes", [])
            for idx, bbox in enumerate(bboxes):
                if bbox.get("label") != "picture":
                    continue
                area = bbox_area(bbox)
                if area < ICON_AREA_MIN or area > ICON_AREA_MAX:
                    continue

                on_legend = (doc_name, page["page_num"]) in legend_keys

                img = crop_bbox_from_pdf(game_id, doc_name, page, bbox)
                if img is None:
                    continue

                dhash = compute_dhash(img)

                candidates.append(IconCandidate(
                    game_id=game_id,
                    doc_name=doc_name,
                    page_num=page["page_num"],
                    bbox_index=idx,
                    bbox=bbox,
                    area=area,
                    image=img,
                    dhash=dhash,
                    on_legend_page=on_legend,
                ))

    if on_progress:
        on_progress(f"Inventoried {len(candidates)} icon candidates")
    return candidates


def _deduplicate_icons(candidates: list[IconCandidate]) -> dict[str, list[IconCandidate]]:
    """Group candidates by DHash similarity into clusters."""
    clusters: dict[str, list[IconCandidate]] = {}
    assigned: set[int] = set()

    for i, c in enumerate(candidates):
        if i in assigned:
            continue

        cluster_id = c.dhash
        clusters[cluster_id] = [c]
        assigned.add(i)

        for j in range(i + 1, len(candidates)):
            if j in assigned:
                continue
            dist = hamming_distance(c.dhash, candidates[j].dhash)
            if dist <= ICON_HASH_MATCH_THRESHOLD:
                clusters[cluster_id].append(candidates[j])
                assigned.add(j)

    return clusters


# ── Stage 3: Resolve Meanings ────────────────────────────────────────────────


def _link_spatial(
    icon_bbox: dict[str, Any],
    all_bboxes: list[dict[str, Any]],
    max_distance: float = 100.0,
) -> str | None:
    """Find the nearest non-picture text bbox by centroid distance with direction bias."""
    icon_cx = (icon_bbox["x0"] + icon_bbox["x1"]) / 2
    icon_cy = (icon_bbox["y0"] + icon_bbox["y1"]) / 2

    best_text = None
    best_score = float("inf")

    for b in all_bboxes:
        if b.get("label") == "picture" or not b.get("text", "").strip():
            continue

        b_cx = (b["x0"] + b["x1"]) / 2
        b_cy = (b["y0"] + b["y1"]) / 2

        dx = b_cx - icon_cx
        dy = b_cy - icon_cy  # In Docling coords: positive = above, negative = below

        raw_dist = math.sqrt(dx * dx + dy * dy)
        if raw_dist > max_distance:
            continue

        # Directional bias: prefer below (dy < 0) and right (dx > 0).
        # Penalize above and left.
        penalty = 1.0
        if dy > 0:  # text is above icon in Docling coords
            penalty *= 1.5
        if dx < 0:  # text is to the left
            penalty *= 1.5

        score = raw_dist * penalty
        if score < best_score:
            best_score = score
            best_text = b.get("text", "").strip()

    return best_text


def _resolve_from_legends(
    clusters: dict[str, list[IconCandidate]],
    all_pages: dict[str, list[dict[str, Any]]],
) -> tuple[list[GlossaryEntry], dict[str, list[IconCandidate]]]:
    """Resolve icon meanings from legend pages using spatial linking."""
    entries: list[GlossaryEntry] = []
    unresolved: dict[str, list[IconCandidate]] = {}

    for cluster_id, icons in clusters.items():
        legend_icons = [ic for ic in icons if ic.on_legend_page]
        if not legend_icons:
            unresolved[cluster_id] = icons
            continue

        # Use the first legend icon to find the meaning via spatial linking.
        li = legend_icons[0]
        page_data = None
        for page in all_pages.get(li.doc_name, []):
            if page["page_num"] == li.page_num:
                page_data = page
                break

        if page_data is None:
            unresolved[cluster_id] = icons
            continue

        meaning = _link_spatial(li.bbox, page_data.get("bboxes", []))
        if not meaning or len(meaning) < 2:
            unresolved[cluster_id] = icons
            continue

        entry = GlossaryEntry(
            id=f"icon_{uuid.uuid4().hex[:8]}",
            name=meaning[:60],
            meaning=meaning,
            source="legend",
            source_detail={
                "doc_name": li.doc_name,
                "page_num": li.page_num,
                "bbox_index": li.bbox_index,
            },
            dhash=cluster_id,
            occurrences=[
                {"doc_name": ic.doc_name, "page_num": ic.page_num, "bbox_index": ic.bbox_index}
                for ic in icons
            ],
            confidence=1.0,
        )
        entries.append(entry)

    return entries, unresolved


def _resolve_via_vlm(
    unresolved: dict[str, list[IconCandidate]],
    known_entries: list[GlossaryEntry],
    all_pages: dict[str, list[dict[str, Any]]],
    game_id: str,
    on_progress: Any = None,
) -> tuple[list[GlossaryEntry], list[dict[str, Any]]]:
    """Use a VLM to infer meanings for unresolved icons, batched by page."""
    from boardgame_agent.config import ANTHROPIC_API_KEY, MODEL_OPTIONS

    provider = MODEL_OPTIONS.get(GLOSSARY_VLM_MODEL, "anthropic")

    # Group unresolved icons by (doc_name, page_num) for batching.
    page_groups: dict[tuple[str, int], list[tuple[str, IconCandidate]]] = defaultdict(list)
    for cluster_id, icons in unresolved.items():
        representative = icons[0]
        key = (representative.doc_name, representative.page_num)
        page_groups[key].append((cluster_id, representative))

    known_summary = "\n".join(
        f"- {e.name}: {e.meaning}" for e in known_entries[:30]
    )

    new_entries: list[GlossaryEntry] = []
    still_unresolved: list[dict[str, Any]] = []

    for (doc_name, page_num), icon_pairs in page_groups.items():
        page_data = None
        for page in all_pages.get(doc_name, []):
            if page["page_num"] == page_num:
                page_data = page
                break
        if page_data is None:
            for cid, ic in icon_pairs:
                still_unresolved.append({
                    "doc_name": doc_name, "page_num": page_num,
                    "bbox_index": ic.bbox_index, "dhash": cid,
                    "vlm_description": ic.bbox.get("text", ""),
                })
            continue

        # Render the page for VLM analysis.
        page_png = render_page_for_vlm(game_id, doc_name, page_data)
        if page_png is None:
            for cid, ic in icon_pairs:
                still_unresolved.append({
                    "doc_name": doc_name, "page_num": page_num,
                    "bbox_index": ic.bbox_index, "dhash": cid,
                    "vlm_description": ic.bbox.get("text", ""),
                })
            continue

        icon_descriptions = "\n".join(
            f"- Icon at bbox {ic.bbox_index}: {ic.bbox.get('text', 'no description')}"
            for _, ic in icon_pairs
        )

        prompt = (
            f"This is a page from a board game rulebook ({doc_name}, page {page_num}).\n\n"
            f"I've already identified these icons and their meanings:\n{known_summary}\n\n"
            f"Please identify the meaning of these unresolved icons on this page:\n{icon_descriptions}\n\n"
            f"For each icon, provide:\n"
            f"- A short name (1-3 words)\n"
            f"- Its game-specific meaning (what it represents or triggers in the game)\n\n"
            f"Respond in JSON: [{{'bbox_index': N, 'name': '...', 'meaning': '...'}}]"
        )

        try:
            vlm_response = _call_vlm(prompt, page_png, provider)
            parsed = _parse_vlm_response(vlm_response)

            for item in parsed:
                bidx = item.get("bbox_index")
                matching_pair = next(
                    ((cid, ic) for cid, ic in icon_pairs if ic.bbox_index == bidx),
                    None,
                )
                if matching_pair is None:
                    continue
                cid, ic = matching_pair
                meaning = item.get("meaning", "")
                name = item.get("name", meaning[:30])

                if not meaning or len(meaning) < 3:
                    still_unresolved.append({
                        "doc_name": doc_name, "page_num": page_num,
                        "bbox_index": bidx, "dhash": cid,
                        "vlm_description": ic.bbox.get("text", ""),
                    })
                    continue

                all_icons_in_cluster = unresolved.get(cid, [ic])
                entry = GlossaryEntry(
                    id=f"icon_{uuid.uuid4().hex[:8]}",
                    name=name,
                    meaning=meaning,
                    source="vlm",
                    source_detail={
                        "doc_name": doc_name, "page_num": page_num,
                        "bbox_index": bidx,
                    },
                    dhash=cid,
                    occurrences=[
                        {"doc_name": i.doc_name, "page_num": i.page_num, "bbox_index": i.bbox_index}
                        for i in all_icons_in_cluster
                    ],
                    confidence=0.7,
                )
                new_entries.append(entry)

        except Exception as e:
            if on_progress:
                on_progress(f"VLM error on {doc_name} p{page_num}: {e}")
            for cid, ic in icon_pairs:
                still_unresolved.append({
                    "doc_name": doc_name, "page_num": page_num,
                    "bbox_index": ic.bbox_index, "dhash": cid,
                    "vlm_description": ic.bbox.get("text", ""),
                })

    return new_entries, still_unresolved


def _call_vlm(prompt: str, image_png: bytes, provider: str) -> str:
    """Call a vision-capable LLM with an image and text prompt."""
    import base64

    if provider == "anthropic":
        import anthropic
        from boardgame_agent.config import ANTHROPIC_API_KEY

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        img_b64 = base64.standard_b64encode(image_png).decode()
        response = client.messages.create(
            model=GLOSSARY_VLM_MODEL,
            max_tokens=2048,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return response.content[0].text

    elif provider == "openai":
        import openai
        from boardgame_agent.config import OPENAI_API_KEY

        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        img_b64 = base64.standard_b64encode(image_png).decode()
        response = client.chat.completions.create(
            model=GLOSSARY_VLM_MODEL,
            max_tokens=2048,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return response.choices[0].message.content

    elif provider == "together":
        import openai
        from boardgame_agent.config import TOGETHER_API_KEY

        client = openai.OpenAI(
            api_key=TOGETHER_API_KEY,
            base_url="https://api.together.xyz/v1",
        )
        img_b64 = base64.standard_b64encode(image_png).decode()
        response = client.chat.completions.create(
            model=GLOSSARY_VLM_MODEL,
            max_tokens=2048,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return response.choices[0].message.content

    raise ValueError(f"Unsupported VLM provider: {provider}")


def _parse_vlm_response(text: str) -> list[dict[str, Any]]:
    """Parse VLM JSON response, tolerating markdown fencing."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        return [result]
    except json.JSONDecodeError:
        return []


# ── Stage 4: Review ──────────────────────────────────────────────────────────


def _add_clip_embeddings(entries: list[GlossaryEntry]) -> None:
    """Compute CLIP embeddings for all entries that have images."""
    # We use the representative icon's image for CLIP embedding.
    # Since we've already cropped all candidates, we can't easily
    # go back — so we compute text-based CLIP embeddings from the meaning.
    from boardgame_agent.glossary.image_utils import clip_text_embedding

    for entry in entries:
        if entry.meaning:
            try:
                entry.clip_embedding = clip_text_embedding(
                    f"{entry.name}: {entry.meaning}"
                )
            except Exception:
                pass


# ── Main Pipeline ────────────────────────────────────────────────────────────


def build_glossary(
    game_id: str,
    on_progress: Any = None,
) -> Glossary:
    """Build an icon glossary for a game from all its extracted documents.

    *on_progress* is an optional callback ``(message: str) -> None`` for UI updates.
    """
    from boardgame_agent.db.games import get_documents
    from boardgame_agent.config import GAMES_DB_PATH

    if on_progress:
        on_progress("Loading extracted documents...")

    # Load all extracted pages grouped by doc_name.
    docs = get_documents(game_id, GAMES_DB_PATH)
    all_pages: dict[str, list[dict]] = {}
    for d in docs:
        pages = load_cached_pages(game_id, d["doc_name"])
        if pages:
            all_pages[d["doc_name"]] = pages

    if not all_pages:
        return Glossary(
            game_id=game_id,
            built_at=datetime.now(timezone.utc).isoformat(),
        )

    # Stage 1: Detect legend/reference pages.
    if on_progress:
        on_progress("Detecting legend and icon reference pages...")

    icon_ref_docs = _find_icon_reference_docs(game_id)
    all_flat_pages = [p for pages in all_pages.values() for p in pages]

    # Treat all pages from icon reference docs as legend pages.
    legend_pages = []
    for doc_name in icon_ref_docs:
        legend_pages.extend(all_pages.get(doc_name, []))

    # Also detect legend pages heuristically from other docs.
    non_ref_pages = [
        p for p in all_flat_pages
        if p.get("doc_name") not in icon_ref_docs
    ]
    legend_pages.extend(_detect_legend_pages(non_ref_pages))

    if on_progress:
        on_progress(f"Found {len(legend_pages)} legend/reference pages")

    # Stage 2: Build icon inventory.
    if on_progress:
        on_progress("Cropping and hashing icons...")

    candidates = _inventory_icons(game_id, all_pages, legend_pages, on_progress)
    clusters = _deduplicate_icons(candidates)

    if on_progress:
        on_progress(f"Found {len(clusters)} unique icon clusters from {len(candidates)} instances")

    # Stage 3: Resolve meanings.
    if on_progress:
        on_progress("Resolving icon meanings from legends...")

    entries, unresolved = _resolve_from_legends(clusters, all_pages)

    if on_progress:
        on_progress(f"Legend-resolved: {len(entries)} icons. Unresolved: {len(unresolved)} clusters")

    # VLM resolution — only for icons on legend/reference pages where the
    # layout implies the meaning is visually present. VLM on arbitrary pages
    # would just be guessing — those icons stay unresolved for hash-matching.
    still_unresolved_list: list[dict] = []
    legend_keys = {(p["doc_name"], p["page_num"]) for p in legend_pages}

    # Split unresolved into legend-page icons (VLM can help) vs others (stay unresolved).
    vlm_candidates: dict[str, list[IconCandidate]] = {}
    for cluster_id, icons in unresolved.items():
        legend_icons = [ic for ic in icons if (ic.doc_name, ic.page_num) in legend_keys]
        if legend_icons:
            vlm_candidates[cluster_id] = icons  # Use legend icon for VLM, track all
        else:
            # No legend page occurrence — stays unresolved.
            rep = icons[0]
            still_unresolved_list.append({
                "doc_name": rep.doc_name, "page_num": rep.page_num,
                "bbox_index": rep.bbox_index, "dhash": cluster_id,
                "vlm_description": rep.bbox.get("text", ""),
            })

    if vlm_candidates:
        if on_progress:
            on_progress(f"Using VLM on {len(vlm_candidates)} icon clusters from legend/reference pages...")

        vlm_entries, vlm_unresolved = _resolve_via_vlm(
            vlm_candidates, entries, all_pages, game_id, on_progress
        )
        entries.extend(vlm_entries)
        still_unresolved_list.extend(vlm_unresolved)

        if on_progress:
            on_progress(
                f"VLM resolved: {len(vlm_entries)} icons. "
                f"Still unresolved: {len(still_unresolved_list)}"
            )
    elif unresolved and on_progress:
        on_progress(f"{len(still_unresolved_list)} icons unresolved (no legend page to reference)")

    # Stage 4: Add CLIP embeddings for semantic search.
    if on_progress:
        on_progress("Computing CLIP embeddings...")

    _add_clip_embeddings(entries)

    # Stage 5: Save (only if we resolved at least one icon).
    glossary = Glossary(
        game_id=game_id,
        built_at=datetime.now(timezone.utc).isoformat(),
        entries=entries,
        unresolved=still_unresolved_list,
    )

    save_path = DATA_DIR / "games" / game_id / "glossary.json"
    if entries:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(glossary.model_dump_json(indent=2), encoding="utf-8")
        if on_progress:
            on_progress(
                f"Glossary saved: {len(entries)} icons resolved, "
                f"{len(still_unresolved_list)} unresolved"
            )
    else:
        # Don't save a useless glossary. Remove stale one if it exists.
        if save_path.exists():
            save_path.unlink()
        if on_progress:
            on_progress(
                f"No icons could be resolved ({len(still_unresolved_list)} candidates found "
                f"but none matched to definitions). Glossary not saved."
            )

    return glossary


def load_glossary(game_id: str) -> Glossary | None:
    """Load a previously built glossary from disk."""
    path = DATA_DIR / "games" / game_id / "glossary.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return Glossary(**data)
