# TODO

## Cell image outputs (active)

Design and phased plan:

- **[design/cell-image-outputs.md](design/cell-image-outputs.md)**

Summary direction:

1. **Phase 1** — Text-first execute/read (image placeholders only) ✅
2. **Phase 2** — `read_cell_image` → resized MCP `ImageContent` (opt-in vision) ✅
3. **Phase 3** — Server-managed artifact `path` delivery (no agent-chosen paths)
4. **Phase 4** — MCP Resources delivery (best-effort; host support varies)
5. **Phase 5** — Docs / polish

Do not ship full base64 plots on every `execute_cell` by default.

## Original agent report (problem evidence)

The notes below are a session report that motivated the design (Grok + this MCP): images often dropped/truncated; tables OK as text; plots unreliable unless saved and re-read.

<details>
<summary>Session notes</summary>

Plot images from `plt.show()` are sometimes exposed as image payloads but often fail (`image_dropped_notice` / truncated / integrity). When an image arrives intact it can be inspected; otherwise only the fact that a plot was produced is known.

Workaround that worked better: `fig.savefig(...)` then read the file, or print key stats.

Also weak: rich HTML tables, `display(Markdown)` fidelity — tracked as backlog in the design doc, not Phase 1–2.

</details>
