"""view_page tool — last-resort visual page analysis via VLM."""

from __future__ import annotations

from langchain_core.tools import tool

from boardgame_agent.config import DATA_DIR


def make_page_vision_tool(game_id: str, config: dict | None = None):
    """Return a view_page tool bound to *game_id*.

    Gated at call time via ``config["enable_page_vision"]`` so it can be
    toggled mid-conversation without rebuilding the agent.
    """

    @tool
    def view_page(doc_name: str, page_num: int, question: str) -> str:
        """Visually analyze a page to understand its layout, icons, or visual content.

        Use this when you found a page via search but can't understand it
        from the extracted text alone (e.g. icon-heavy pages, visual layouts).
        The result helps you understand what to search for next — it does NOT
        replace searching for the actual rules. Always follow up by searching
        for the terms and concepts the vision analysis reveals.

        Args:
            doc_name: The document name (as shown in search results).
            page_num: The page number to view.
            question: What you want to understand about this page.
        """
        if config is not None and not config.get("enable_page_vision", False):
            return "Page vision is disabled. Enable it in the sidebar to use this tool."
        from boardgame_agent.config import MODEL_OPTIONS, PAGE_VISION_MODEL
        from boardgame_agent.glossary.builder import _call_vlm
        from boardgame_agent.glossary.image_utils import render_page_for_vlm
        from boardgame_agent.rag.extractor import load_cached_pages

        pages = load_cached_pages(game_id, doc_name)
        if pages is None:
            return f"Document '{doc_name}' not found or not yet extracted."

        page_data = next((p for p in pages if p["page_num"] == page_num), None)
        if page_data is None:
            return f"Page {page_num} not found in '{doc_name}'."

        page_png = render_page_for_vlm(game_id, doc_name, page_data)
        if page_png is None:
            return f"Could not render page {page_num} of '{doc_name}'."

        provider = MODEL_OPTIONS.get(PAGE_VISION_MODEL, "anthropic")

        prompt = (
            f"You are analyzing page {page_num} of a board game rulebook "
            f"document called '{doc_name}'.\n\n"
            f"Question: {question}\n\n"
            f"Describe what you see that answers this question. Be specific "
            f"about any icons, symbols, numbers, or game components visible."
        )

        try:
            return _call_vlm(prompt, page_png, provider)
        except Exception as e:
            return f"Vision analysis failed: {e}"

    return view_page
