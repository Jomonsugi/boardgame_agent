"""lookup_glossary tool — semantic search over the game's icon glossary."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from langchain_core.tools import tool

from boardgame_agent.config import DATA_DIR


def make_glossary_tool(game_id: str):
    """Return a lookup_glossary tool bound to *game_id*."""

    @tool
    def lookup_glossary(query: str) -> str:
        """Look up icons or symbols in the game's icon glossary.

        Use this when you encounter an icon or symbol reference you don't
        understand, or when the user asks about a specific icon.

        Args:
            query: Description of the icon or symbol to look up
                   (e.g. "red star", "shield icon", "fire element").
        """
        from boardgame_agent.glossary.builder import load_glossary

        glossary = load_glossary(game_id)
        if glossary is None or not glossary.entries:
            return "No icon glossary has been built for this game yet."

        # Semantic search using CLIP text embeddings.
        try:
            from boardgame_agent.glossary.image_utils import clip_text_embedding
            query_emb = np.array(clip_text_embedding(query))

            scored = []
            for entry in glossary.entries:
                if not entry.clip_embedding:
                    continue
                entry_emb = np.array(entry.clip_embedding)
                sim = float(np.dot(query_emb, entry_emb))
                scored.append((sim, entry))

            scored.sort(key=lambda x: x[0], reverse=True)
            top = scored[:5]
        except Exception:
            # Fall back to simple text matching if CLIP fails.
            q_lower = query.lower()
            top = [
                (1.0, entry)
                for entry in glossary.entries
                if q_lower in entry.name.lower() or q_lower in entry.meaning.lower()
            ][:5]

        if not top:
            return (
                f"No matching icons found for '{query}'. "
                f"The glossary has {len(glossary.entries)} entries. "
                f"Try a different description."
            )

        lines = []
        for sim, entry in top:
            conf = f" (confidence: {entry.confidence:.0%})" if entry.confidence < 1.0 else ""
            lines.append(f"- **{entry.name}**{conf}: {entry.meaning}")

            # Provide citable references so the agent can include these in
            # submit_answer citations with proper doc_name, page_num, bbox_indices.
            for occ in entry.occurrences[:3]:
                doc = occ.get("doc_name", "")
                page = occ.get("page_num", "?")
                bidx = occ.get("bbox_index")
                if bidx is not None:
                    lines.append(
                        f"  Citation: doc_name=\"{doc}\", page_num={page}, bbox_indices=[{bidx}]"
                    )
                else:
                    lines.append(
                        f"  Citation: doc_name=\"{doc}\", page_num={page}, bbox_indices=[]"
                    )

        return "Icon glossary results:\n" + "\n".join(lines)

    return lookup_glossary
