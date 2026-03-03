/**
 * Tests for context/archive.ts — archive management and turn summarization.
 */

import { describe, expect, it } from "bun:test";
import type { ContextArchive, Message, ToolResultBlock } from "../types.ts";
import {
	appendToWorkSummary,
	createArchive,
	recordDecision,
	recordFileModification,
	recordResolvedError,
	renderArchive,
	summarizeTurn,
} from "./archive.ts";

// ─── Helpers ──────────────────────────────────────────────────────────────────

function toolUseMsg(id: string, name: string, input: Record<string, unknown>): Message {
	return {
		role: "assistant",
		content: [{ type: "tool_use", id, name, input }],
	};
}

function toolResultMsg(toolUseId: string, content: string, isError = false): Message {
	const block: ToolResultBlock = { type: "tool_result", tool_use_id: toolUseId, content };
	if (isError) block.is_error = true;
	// Cast needed: ToolResultBlock appears in user messages at runtime via loop.ts LoopMessage,
	// but ContentBlock union doesn't include tool_result to avoid burdening other consumers.
	return { role: "user", content: [block as unknown as import("../types.ts").ContentBlock] };
}

function emptyArchive(): ContextArchive {
	return createArchive();
}

// ─── createArchive ─────────────────────────────────────────────────────────────

describe("createArchive", () => {
	it("returns a fresh empty archive", () => {
		const archive = createArchive();
		expect(archive.workSummary).toBe("");
		expect(archive.decisions).toEqual([]);
		expect(archive.modifiedFiles.size).toBe(0);
		expect(archive.fileHashes.size).toBe(0);
		expect(archive.resolvedErrors).toEqual([]);
	});
});

// ─── summarizeTurn ────────────────────────────────────────────────────────────

describe("summarizeTurn", () => {
	it("returns fallback when no tool calls", () => {
		const msgs: Message[] = [{ role: "assistant", content: [{ type: "text", text: "Done." }] }];
		expect(summarizeTurn(1, msgs)).toBe("Turn 1: (no tool calls)");
	});

	it("summarizes read tool call with line count from tool result", () => {
		const msgs: Message[] = [
			toolUseMsg("id1", "read", { file_path: "project/src/foo.ts" }),
			toolResultMsg("id1", "line1\nline2\nline3"),
		];
		const summary = summarizeTurn(2, msgs);
		expect(summary).toContain("Turn 2: read(…/src/foo.ts)");
		expect(summary).toContain("3 lines");
	});

	it("summarizes write tool call as written on success", () => {
		const msgs: Message[] = [
			toolUseMsg("id2", "write", { file_path: "project/src/bar.ts" }),
			toolResultMsg("id2", "File written successfully"),
		];
		const summary = summarizeTurn(3, msgs);
		expect(summary).toContain("write(…/src/bar.ts) → written");
	});

	it("summarizes write tool call as failed on error", () => {
		const msgs: Message[] = [
			toolUseMsg("id3", "write", { file_path: "project/src/baz.ts" }),
			toolResultMsg("id3", "Permission denied error"),
		];
		const summary = summarizeTurn(4, msgs);
		expect(summary).toContain("write(…/src/baz.ts) → failed");
	});

	it("summarizes edit tool call", () => {
		const msgs: Message[] = [
			toolUseMsg("id4", "edit", { file_path: "project/src/index.ts" }),
			toolResultMsg("id4", "Edit applied"),
		];
		const summary = summarizeTurn(5, msgs);
		expect(summary).toContain("edit(…/src/index.ts) → edited");
	});

	it("summarizes bash tool call with ok result", () => {
		const msgs: Message[] = [
			toolUseMsg("id5", "bash", { command: "bun test" }),
			toolResultMsg("id5", "All tests passed"),
		];
		const summary = summarizeTurn(6, msgs);
		expect(summary).toContain("bash(bun test) → ok");
	});

	it("summarizes bash tool call with error exit code", () => {
		const msgs: Message[] = [
			toolUseMsg("id6", "bash", { command: "bun test" }),
			toolResultMsg("id6", "Tests failed with exit code 1"),
		];
		const summary = summarizeTurn(7, msgs);
		expect(summary).toContain("bash(bun test) → error");
	});

	it("summarizes grep tool call with match count", () => {
		const msgs: Message[] = [
			toolUseMsg("id7", "grep", { pattern: "findToolResult" }),
			toolResultMsg("id7", "5 matches found in 3 files"),
		];
		const summary = summarizeTurn(8, msgs);
		expect(summary).toContain("grep(findToolResult) → 5 matches");
	});

	it("summarizes glob tool call with file count", () => {
		const msgs: Message[] = [
			toolUseMsg("id8", "glob", { pattern: "**/*.ts" }),
			toolResultMsg("id8", "src/a.ts\nsrc/b.ts\nsrc/c.ts"),
		];
		const summary = summarizeTurn(9, msgs);
		expect(summary).toContain("glob(**/*.ts) → 3 files");
	});

	it("summarizes unknown tool call as done when result exists", () => {
		const msgs: Message[] = [
			toolUseMsg("id9", "custom_tool", {}),
			toolResultMsg("id9", "something"),
		];
		const summary = summarizeTurn(10, msgs);
		expect(summary).toContain("custom_tool(…) → done");
	});

	it("shows ? for unknown tool when no result found", () => {
		const msgs: Message[] = [toolUseMsg("id10", "custom_tool", {})];
		const summary = summarizeTurn(10, msgs);
		expect(summary).toContain("custom_tool(…) → ?");
	});

	it("does not match tool result from a different tool_use_id", () => {
		const msgs: Message[] = [
			toolUseMsg("id-a", "read", { file_path: "a.ts" }),
			toolResultMsg("id-b", "wrong result"), // different ID
		];
		const summary = summarizeTurn(1, msgs);
		// Without a matching result, line count is "?"
		expect(summary).toContain("? lines");
	});

	it("correctly associates result when multiple tool calls in same turn", () => {
		const msgs: Message[] = [
			{
				role: "assistant",
				content: [
					{ type: "tool_use", id: "id-r1", name: "read", input: { file_path: "a.ts" } },
					{ type: "tool_use", id: "id-r2", name: "read", input: { file_path: "b.ts" } },
				],
			},
			toolResultMsg("id-r1", "line1\nline2"),
			toolResultMsg("id-r2", "x\ny\nz"),
		];
		const summary = summarizeTurn(1, msgs);
		expect(summary).toContain("2 lines");
		expect(summary).toContain("3 lines");
	});
});

// ─── appendToWorkSummary ──────────────────────────────────────────────────────

describe("appendToWorkSummary", () => {
	it("appends to an empty summary", () => {
		const archive = appendToWorkSummary(emptyArchive(), "Turn 1: read(a.ts) → 10 lines", 10000);
		expect(archive.workSummary).toBe("Turn 1: read(a.ts) → 10 lines");
	});

	it("appends newline-separated entries", () => {
		let archive = appendToWorkSummary(emptyArchive(), "entry1", 10000);
		archive = appendToWorkSummary(archive, "entry2", 10000);
		expect(archive.workSummary).toBe("entry1\nentry2");
	});

	it("drops oldest lines when over budget", () => {
		const longEntry = "x".repeat(400); // ~100 tokens
		let archive = emptyArchive();
		for (let i = 0; i < 5; i++) {
			archive = appendToWorkSummary(archive, `entry${i}: ${longEntry}`, 200);
		}
		// Oldest entries should have been dropped
		expect(archive.workSummary).not.toContain("entry0:");
	});
});

// ─── recordFileModification ───────────────────────────────────────────────────

describe("recordFileModification", () => {
	it("adds a file entry to the archive", () => {
		const archive = recordFileModification(emptyArchive(), "src/foo.ts", "added feature X");
		expect(archive.modifiedFiles.get("src/foo.ts")).toBe("added feature X");
	});

	it("updates existing file entry", () => {
		let archive = recordFileModification(emptyArchive(), "src/foo.ts", "first");
		archive = recordFileModification(archive, "src/foo.ts", "updated");
		expect(archive.modifiedFiles.get("src/foo.ts")).toBe("updated");
	});

	it("does not mutate the original archive", () => {
		const original = emptyArchive();
		recordFileModification(original, "src/foo.ts", "change");
		expect(original.modifiedFiles.size).toBe(0);
	});
});

// ─── recordResolvedError ──────────────────────────────────────────────────────

describe("recordResolvedError", () => {
	it("appends an error summary", () => {
		const archive = recordResolvedError(emptyArchive(), "Fixed type mismatch in archive.ts");
		expect(archive.resolvedErrors).toContain("Fixed type mismatch in archive.ts");
	});

	it("does not mutate the original archive", () => {
		const original = emptyArchive();
		recordResolvedError(original, "something");
		expect(original.resolvedErrors).toHaveLength(0);
	});
});

// ─── recordDecision ───────────────────────────────────────────────────────────

describe("recordDecision", () => {
	it("appends a decision", () => {
		const archive = recordDecision(emptyArchive(), "Use tool_result variant in ContentBlock");
		expect(archive.decisions).toContain("Use tool_result variant in ContentBlock");
	});

	it("accumulates multiple decisions", () => {
		let archive = recordDecision(emptyArchive(), "decision A");
		archive = recordDecision(archive, "decision B");
		expect(archive.decisions).toHaveLength(2);
	});
});

// ─── renderArchive ────────────────────────────────────────────────────────────

describe("renderArchive", () => {
	it("returns empty string for empty archive", () => {
		expect(renderArchive(emptyArchive())).toBe("");
	});

	it("renders work summary section", () => {
		const archive = appendToWorkSummary(emptyArchive(), "Turn 1: read(a.ts) → 5 lines", 10000);
		const rendered = renderArchive(archive);
		expect(rendered).toContain("## Work So Far");
		expect(rendered).toContain("Turn 1: read(a.ts) → 5 lines");
	});

	it("renders modified files section", () => {
		const archive = recordFileModification(emptyArchive(), "src/foo.ts", "added X");
		const rendered = renderArchive(archive);
		expect(rendered).toContain("## Files Modified");
		expect(rendered).toContain("src/foo.ts: added X");
	});

	it("renders decisions section", () => {
		const archive = recordDecision(emptyArchive(), "use tabs");
		const rendered = renderArchive(archive);
		expect(rendered).toContain("## Key Decisions");
		expect(rendered).toContain("use tabs");
	});

	it("renders resolved errors section", () => {
		const archive = recordResolvedError(emptyArchive(), "Fixed lint error");
		const rendered = renderArchive(archive);
		expect(rendered).toContain("## Resolved Issues");
		expect(rendered).toContain("Fixed lint error");
	});

	it("renders all sections when all are populated", () => {
		let archive = emptyArchive();
		archive = appendToWorkSummary(archive, "Turn 1: bash(bun test) → ok", 10000);
		archive = recordFileModification(archive, "src/a.ts", "fix");
		archive = recordDecision(archive, "use bun");
		archive = recordResolvedError(archive, "test fixed");
		const rendered = renderArchive(archive);
		expect(rendered).toContain("## Work So Far");
		expect(rendered).toContain("## Files Modified");
		expect(rendered).toContain("## Key Decisions");
		expect(rendered).toContain("## Resolved Issues");
	});
});
