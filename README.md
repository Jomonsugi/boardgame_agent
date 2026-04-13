# Board Game Rules Agent

A local AI assistant that answers board game rules questions with **cited, highlighted** references to the official rulebook — built for fast lookups during actual gameplay.

Ask a question, get an answer with clickable citations that highlight the exact source text in the PDF viewer. The agent cross-references multiple documents when needed and keeps digging until it can ground every claim.

## Quick start

Prerequisites:
- [uv](https://docs.astral.sh/uv/getting-started/installation/): `curl -LsSf https://astral.sh/uv/install.sh | sh`
- [Ollama](https://ollama.com/download): download the macOS app, then `ollama pull qwen3-embedding`
- A [Together API](https://www.together.ai/) key (free tier works — default LLM provider)

```bash
cd boardgame_agent
uv sync
cp .env.example .env
# Edit .env and add your TOGETHER_API_KEY
boardgame-agent
```

### Optional API keys

Add these to `.env` for additional features:

| Key | Purpose | Free tier? |
|-----|---------|------------|
| `TOGETHER_API_KEY` | Default LLM provider | No |
| `ANTHROPIC_API_KEY` | Claude models (agent or VLM) | No |
| `OPENAI_API_KEY` | GPT-4o models | No |
| `COHERE_API_KEY` | Re-ranking (improves retrieval accuracy) | Yes (1k calls/month) |
| `TAVILY_API_KEY` | Web search fallback | Yes |

---

## Setting up a game

### 1. Create a game

Click **Add new game** in the sidebar. The new game is auto-selected.

### 2. Upload documents

Upload your rulebook PDF and any supplemental documents (FAQ, player aids, icon references, logbooks). Each document gets:

- **Tag** — auto-suggested from the filename (e.g., "Icon-Overview.pdf" suggests `icon_reference`). Edit anytime — changes apply instantly, no reindexing.
- **Description** — optional, helps the agent decide when to search this document. Example: "Contains all 50 mission criteria and special rules per mission."

### 3. Processing options (on by default)

When uploading PDFs, two options are checked by default:

- **Enrich pictures with VLM descriptions** — a local vision model (Qwen2.5-VL 3B) describes every icon, symbol, and diagram in the PDF. This makes visual elements searchable. Uncheck for text-only rulebooks where icons aren't important.
- **Build icon glossary after indexing** — automatically builds a structured glossary that maps icons to their game-specific meanings by detecting legend pages and cross-referencing icon definitions. Uncheck for simple games without meaningful iconography.

Both can be done later from the sidebar if you skip them during upload.

### 4. Ask questions

Type a rules question in the chat. The agent searches, cross-references when needed, and returns a cited answer. Click any **citation chip** to view the source with highlighted text in the PDF viewer.

---

## How the agent reasons

The agent follows a natural reasoning loop — the same process a human uses when looking up a rule:

1. **Search** the most relevant document for the question
2. **Evaluate** — "Can I fully answer now? Is there anything in these results I don't understand?"
3. **If yes** — submit the answer with citations immediately
4. **If no** — search for the unknown terms, icons, or mechanics, then go back to step 2

This means simple questions ("How many cards do you draw?") are answered in one search. Complex questions that involve icons, cross-document references, or interacting rules trigger multiple searches automatically — the agent keeps going until every part of the answer is grounded.

### Cross-referencing

When the agent finds information in a supplement or logbook that references game mechanics defined in the rulebook, it automatically cross-references the rulebook before answering. The final answer cites all sources that contributed.

---

## Features

### Citations

Every answer includes clickable citation chips showing document name and page number. Click a citation to view the PDF page with highlighted bounding boxes around the cited text. Citations come from text retrieval — the agent must find and cite the actual rules, not guess.

### Icon glossary

For games with meaningful icons (The Crew, Ark Nova, Gloomhaven), the glossary builder:

- Detects legend/reference pages automatically (heuristic scoring)
- Links icons to adjacent text labels using spatial proximity
- Deduplicates icons across all documents using perceptual hashing (DHash)
- Resolves unmatched icons on legend pages using a vision model
- Makes icon meanings searchable via the `lookup_glossary` tool

View, rebuild, or reindex with glossary enrichment from the sidebar.

### Re-ranking

After hybrid retrieval (dense + sparse vectors with RRF fusion), results are re-ranked with a cross-encoder for higher precision. Default: Cohere Rerank API (free tier). Falls back to local FastEmbed if no API key is set.

### Web search

When all indexed documents have been exhausted and the answer is still unclear, the agent can search the web. Restricted to trusted domains you configure per game (default: boardgamegeek.com). Requires a Tavily API key.

### Answer history

Rate answers with thumbs up/down. Accepted answers feed into the `get_past_answers` tool so the agent stays consistent with prior verified rulings.

### Page vision

The agent can visually analyze a page when text extraction doesn't capture enough (icon-heavy pages, complex layouts). The vision model helps the agent understand what to search for next — it doesn't replace text retrieval.

---

## Document options

These are available per document in the sidebar under **Options**:

- **Description** — helps the agent choose which document to search
- **Two-page spreads** — splits landscape pages into left/right halves
- **Picture enrichment** — re-run VLM description with a different model
- **Tag** — editable inline, changes apply instantly

---

## LLM providers

Models and their providers are configured in `config.py` under `MODEL_OPTIONS`. Map each model ID to `"together"`, `"anthropic"`, or `"openai"`. Only add API keys for the providers you use. Switch models from the sidebar dropdown — changing the model resets the conversation.

## Embeddings

Dense vectors via Ollama (default `qwen3-embedding`, 4096-d). Sparse vectors via FastEmbed SPLADE++. Results fused with Qdrant-native RRF hybrid search. Change `OLLAMA_EMBED_MODEL` in `config.py` and click **Rebuild index** in the sidebar.

Ollama launches automatically if installed but not running.

## Supported document formats

- **PDF** — parsed by Docling with bounding-box citations and highlighted page rendering
- **Markdown** (.md) — parsed by heading structure with text-based citation highlighting
