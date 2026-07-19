# Cell image outputs for agent-friendly MCP

**Status:** design (not implemented)  
**Date:** 2026-07-19  
**Repo:** [sglkc/jupyter-mcp-server](https://github.com/sglkc/jupyter-mcp-server) (fork of [datalayer/jupyter-mcp-server](https://github.com/datalayer/jupyter-mcp-server))  
**Related upstream:** [#200](https://github.com/datalayer/jupyter-mcp-server/issues/200) (Resources / size), [#214](https://github.com/datalayer/jupyter-mcp-server/issues/214) (structured_output / rendering, fixed), [#275](https://github.com/datalayer/jupyter-mcp-server/issues/275) (stream + ImageContent, fixed)

## Problem

Agents need to *see* notebook plots (matplotlib, seaborn, PIL, etc.), but shipping full base64 PNGs on every `execute_cell` / `read_cell` is unreliable and expensive:

1. **Payload size** — typical plots produce 70k–100k+ base64 characters; clients truncate or drop tool results (`image_dropped_notice`, integrity failures, “saved to temp file”).
2. **Token waste** — images on every execute force vision cost even when the model only needed stdout/tables.
3. **Not re-readable** — images gated only to the execute response cannot be inspected later without re-running the cell.
4. **Filesystem confusion** — “save to path” is not portable: MCP may run in a Jupyter container while the agent reads a different machine’s disk.

MCP itself defines **no max size** for `ImageContent`. Limits are client/host-specific (e.g. Claude Code `MAX_MCP_OUTPUT_TOKENS` default **25 000**). There is no single safe universal byte limit; resize/compress on the server is the only portable control.

## Current behavior (upstream baseline)

| Piece | Behavior |
|--------|----------|
| `extract_output` (`utils.py`) | `image/png` → `ImageContent` when `ALLOW_IMG_OUTPUT=true` |
| MIME coverage | PNG only; jpeg/svg/gif ignored; HTML stubbed as `[HTML Output]` |
| Tools | `execute_cell`, `insert_execute_code_cell`, `read_cell`, `execute_code` return `list[str \| ImageContent]` |
| `structured_output=False` | Already set (fixes clients dumping base64 as text) |
| MCP Resources | Effectively unused (`resources/list` empty in extension path) |
| Tests | Multimodal PNG generation covered; stream image drain covered after #275 |

Inline images *work in tests* but fail under real agent hosts for large plots.

## Goals

1. Agents can inspect **any** stored cell image (not only the last execute).
2. **Default execute/read paths stay text-cheap** (placeholders, not pixels).
3. Images are **opt-in** via a dedicated tool.
4. Delivery modes work for **local stdio** and **remote Jupyter** without requiring the agent to invent paths.
5. Always **resize/compress** before leaving the server.
6. Document limits, modes, and failure modes for operators and agents.

## Non-goals (for initial phases)

- Perfect rich HTML / widget fidelity (separate problem; see backlog).
- Full MCP Resource client compatibility (blocked by host support; #200 prototype failed on Claude Code).
- Writing into a proprietary “agent session directory” without host cooperation (not a server-owned concept).

## Design decisions

### D1 — Dedicated tool, not “always inline”

**New tool:** `read_cell_image` (name finalizable at implementation).

- Reads image outputs already stored on a code cell (after execute or already in the notebook).
- Args: `cell_index`, `output_index` or `image_index`, optional `max_edge` / quality.
- Not gated on “you just executed.”

**Rationale:** lower default token cost; re-read existing plots; model chooses when to pay for vision.

### D2 — Execute / read return placeholders for images

When a cell has image mime data, text tools return a short placeholder, e.g.:

```text
[image output #0: image/png ~1240x780 ~92KB — use read_cell_image(cell_index=…, image_index=0)]
```

No base64 in `execute_cell` / `read_cell` / `insert_execute_code_cell` by default.

### D3 — Server-owned delivery modes (no free-form agent paths as primary API)

```text
read_cell_image(..., delivery="image" | "resource" | "path")
```

| `delivery` | Behavior | When it works |
|------------|----------|----------------|
| **`image`** (default) | Resize → return MCP `ImageContent` | Any host with multimodal tool results; works remote |
| **`resource`** | Resize → server cache → resource URI | Hosts that implement Resources |
| **`path`** | Resize → write under **server-configured artifact root** → return resolved path only | Colocated FS (stdio + roots / `JUPYTER_MCP_ARTIFACT_DIR`) |

- Agent does **not** choose arbitrary paths by default.
- Optional advanced override (`save_path` under allowlist) only if needed later.
- Artifact layout example: `{artifact_root}/{notebook_id}/cell-{i}-img-{j}.png`

### D4 — Shared image pipeline

Single helper used by all deliveries:

1. Locate notebook cell output (nbformat / YDoc / contents).
2. Prefer `image/png`, then `image/jpeg`, `image/gif` (svg later if needed).
3. Decode → resize (max edge default **1024**) → re-encode (PNG or JPEG).
4. Enforce soft max raw size; re-compress or error with a clear message.
5. Deliver per mode.

### D5 — What “session directory” means

The agent host owns session/artifact directories. The server only:

- writes under **Roots** or `JUPYTER_MCP_ARTIFACT_DIR` when configured, or  
- returns bytes/`ImageContent` / resource URIs and lets the **host** materialize files.

Never claim remote MCP can write the agent laptop’s session folder without a shared volume or host-side materialization.

## Phased plan

### Phase 0 — Baseline & docs (this document)

- [x] Capture problem, limits, decisions, phases.
- [ ] Link from `TODO.md` / contributor notes as needed.
- [ ] Keep `ALLOW_IMG_OUTPUT` behavior documented until Phase 1 lands.

### Phase 1 — Text-first outputs + shared extract changes

**Scope**

- Change `extract_output` / call sites so execution and read tools emit **image placeholders** instead of full `ImageContent` by default (or via a clear config defaulting to off for bulk tools).
- Preserve ability to detect *that* an image exists (index, mime, approximate size/dims if cheap).
- Extend MIME recognition at least to jpeg/gif for *detection* (full decode in Phase 2).
- Tests: execute plot cell → placeholder text, no huge base64; non-image outputs unchanged.

**Success**

- Tool results for plots stay small and stable under Claude Code / Grok-style hosts.
- Agents can discover image indices from `execute_cell` / `read_cell`.

### Phase 2 — `read_cell_image` with `delivery="image"`

**Scope**

- New tool registered next to cell tools.
- Load cell output → resize/compress → return single `ImageContent` (+ short metadata text).
- `structured_output=False` on the tool.
- Config: `max_edge`, quality, maybe max bytes.
- Works for both `MCP_SERVER` and `JUPYTER_SERVER` modes (same notebook access patterns as `read_cell`).
- Tests: synthetic PNG in cell → tool returns `ImageContent` under size budget; missing index → clear error; multi-image cells select by index.

**Success**

- Agent can opt into vision for one plot without re-executing and without paying on every execute.

### Phase 3 — Server-managed `delivery="path"`

**Scope**

- Artifact root from env (`JUPYTER_MCP_ARTIFACT_DIR`) and/or MCP Roots when available.
- Write compressed file; tool returns path + metadata only (`include_image` false by default).
- Refuse path mode with a clear error when no writable root is configured (typical pure remote HTTP without shared FS).
- Optional: ensure `list_files` can see artifacts if under Jupyter contents root.

**Success**

- Local/stdio agents can re-read plots via host file tools without agent-invented paths.

### Phase 4 — `delivery="resource"` (best-effort)

**Scope**

- In-memory or disk cache of exported images.
- Implement `resources/list` + `resources/read` for `jupyter://…` (or similar) URIs.
- Tool returns resource link / lightweight handle.
- Document host support gaps (see #200).

**Success**

- Hosts that support Resources can fetch images without stuffing every tool result; degrade gracefully elsewhere.

### Phase 5 — Polish & docs site

- Operator docs: env vars, defaults, size budgets, remote vs local matrix.
- Agent-facing tool descriptions: when to call `read_cell_image`, how indices work.
- Consider prompt snippets under `prompt/`.
- Optional: config to re-enable legacy “inline images on execute” for niche clients.

## Size budgets (engineering targets, not protocol law)

| Knob | Suggested default | Notes |
|------|-------------------|--------|
| Max edge | 1024 px | Readable axes/labels for most plots |
| Soft max raw bytes after compress | ~100–150 KB | Reduces drop risk |
| Hard fail / re-encode loop | ~200 KB raw | Clear error if still too large |
| Default delivery | `image` | Portable across remote/local |

Token accounting when handled as **true** multimodal images is vision-token based (often ~1–2k for a plot). When mishandled as **text** base64, cost explodes — keep `structured_output=False` and prefer small payloads.

## Delivery mode comparison

| | `ImageContent` | Artifact `path` | MCP Resource |
|--|----------------|-----------------|--------------|
| Agent chooses path? | No | No (server layout) | No |
| Shared disk required? | No | Yes | No (fetch over MCP) |
| Remote Jupyter HTTP | Best default | Often unavailable | Depends on host |
| Re-read later | Via chat history or re-call tool | Via file path | Via resource URI |
| Client maturity | Good if multimodal tools work | Depends on host file tools | Uneven (#200) |

## Configuration sketch

```text
ALLOW_IMG_OUTPUT          # existing; revisit once Phase 1 defaults change
JUPYTER_MCP_ARTIFACT_DIR  # Phase 3: writable root for delivery=path
JUPYTER_MCP_IMAGE_MAX_EDGE=1024
JUPYTER_MCP_IMAGE_MAX_BYTES=150000
```

Exact names finalized in implementation PRs.

## Testing strategy

1. **Unit:** decode/resize/encode helper; placeholder formatting; index selection.
2. **Integration:** insert cell → execute → placeholder → `read_cell_image` → `ImageContent` mime and size bounds.
3. **Mode matrix:** `MCP_SERVER` + `JUPYTER_SERVER` for read path.
4. **Regression:** stream execute still handles non-image outputs; no `ImageContent.strip()` regressions.
5. **Negative:** empty outputs, markdown cells, invalid index, path mode without artifact root.

## Out-of-scope backlog (related agent pain)

| Topic | Notes |
|--------|--------|
| HTML tables / styled DataFrames | Currently weak; often only `text/plain` |
| `display(Markdown)` | Often object repr, not rendered HTML |
| Full notebook dump token bloat | Upstream discussion #169 |
| Client-side Resource bugs | Track hosts; don’t block Phases 1–2 |

## Implementation sketch (files likely touched)

| Area | Files (indicative) |
|------|---------------------|
| Extract / format | `jupyter_mcp_server/utils.py` |
| New tool | `jupyter_mcp_server/tools/read_cell_image_tool.py` |
| Registration | `jupyter_mcp_server/server.py`, `tools/__init__.py` |
| Config | `jupyter_mcp_server/config.py` |
| Extension resources | `jupyter_extension/handlers.py` (Phase 4) |
| Tests | `tests/test_tools.py`, new `tests/test_read_cell_image.py` |
| User docs | `docs/docs/reference/tools/…`, README image section |

## Open questions (resolve during Phase 1–2 PRs)

1. Indexing: **image-only index** vs raw **output list index**? Prefer image-only for agent UX; document both in tool description if both supported.
2. Should `read_cell` with `include_outputs=true` still ever embed images under a flag, or always placeholders after Phase 1?
3. Dependency for resize: Pillow (already used in tests) vs pure stdlib — prefer Pillow optional with clear error if missing?
4. Default `delivery` when artifact root is set — still `image`, or auto-`path`? Prefer **always explicit default `image`** for predictability.

## Phase exit criteria (summary)

| Phase | Exit when |
|-------|-----------|
| 1 | Plot execute results are small placeholders; tests green |
| 2 | `read_cell_image` returns resized `ImageContent` in both server modes |
| 3 | Path delivery works under configured root; fails clearly otherwise |
| 4 | Resources list/read functional; documented host caveats |
| 5 | Published docs + tool descriptions match behavior |

## References

- Agent observation notes: `TODO.md` (session report on image drops)
- Upstream issue #200 — oversized tool results; Resources proposal; prototype failed on Claude Code
- Upstream issue #214 — `structured_output=False` for ImageContent (merged)
- Upstream issue #275 — stream path destroyed ImageContent (merged)
- MCP tools spec — `ImageContent` has no protocol size limit
- MCP Roots — client-advertised filesystem boundaries for server writes
