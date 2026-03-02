/**
 * Working memory / long-term store for the context manager.
 *
 * The archive captures compacted information from pruned turns:
 * - workSummary: rolling summary of work done
 * - decisions: key decisions made by the agent
 * - modifiedFiles: files the agent has modified with brief descriptions
 * - fileHashes: content hashes for staleness detection
 * - resolvedErrors: errors that were encountered and resolved
 */

import type { ContentBlock, ContextArchive, Message } from "../types.ts";
import { estimateTokens } from "./measure.ts";

/**
 * Create a fresh empty archive.
 */
export function createArchive(): ContextArchive {
	return {
		workSummary: "",
		decisions: [],
		modifiedFiles: new Map(),
		fileHashes: new Map(),
		resolvedErrors: [],
	};
}

/**
 * Generate a one-line summary for a turn using template-based approach (no LLM).
 * Format: Turn {N}: {tool_name}({key_args}) → {outcome}
 */
export function summarizeTurn(turnIndex: number, messages: Message[]): string {
	const parts: string[] = [];

	for (const msg of messages) {
		if (msg.role !== "assistant") continue;

		const blocks = typeof msg.content === "string" ? [] : msg.content;
		for (const block of blocks) {
			if (block.type !== "tool_use") continue;

			const summary = summarizeToolCall(block, turnIndex, messages);
			if (summary) parts.push(summary);
		}
	}

	if (parts.length === 0) return `Turn ${turnIndex}: (no tool calls)`;
	return parts.join("; ");
}

/**
 * Summarize a single tool call block into a one-liner.
 */
function summarizeToolCall(
	block: ContentBlock & { type: "tool_use" },
	turnIndex: number,
	allMessages: Message[],
): string {
	const name = block.name;
	const input = block.input;

	// Find the tool result for this tool call
	const outcome = findToolResult(block.id, allMessages);

	switch (name) {
		case "read": {
			const path = typeof input.file_path === "string" ? input.file_path : "file";
			const lineCount = outcome ? countLines(outcome) : "?";
			return `Turn ${turnIndex}: read(${shortPath(path)}) → ${lineCount} lines`;
		}
		case "write": {
			const path = typeof input.file_path === "string" ? input.file_path : "file";
			const isError = outcome?.includes("error") ?? false;
			return `Turn ${turnIndex}: write(${shortPath(path)}) → ${isError ? "failed" : "written"}`;
		}
		case "edit": {
			const path = typeof input.file_path === "string" ? input.file_path : "file";
			const isError = outcome?.includes("error") ?? false;
			return `Turn ${turnIndex}: edit(${shortPath(path)}) → ${isError ? "failed" : "edited"}`;
		}
		case "bash": {
			const cmd = typeof input.command === "string" ? input.command : "cmd";
			const shortCmd = `${cmd.length > 40 ? cmd.slice(0, 40) : cmd}${cmd.length > 40 ? "…" : ""}`;
			const isError = outcome?.match(/exit code [^0]/) ?? false;
			return `Turn ${turnIndex}: bash(${shortCmd}) → ${isError ? "error" : "ok"}`;
		}
		case "grep": {
			const pattern = typeof input.pattern === "string" ? input.pattern : "?";
			const matchCount = outcome ? extractMatchCount(outcome) : "?";
			return `Turn ${turnIndex}: grep(${pattern}) → ${matchCount} matches`;
		}
		case "glob": {
			const pattern = typeof input.pattern === "string" ? input.pattern : "?";
			const fileCount = outcome ? countLines(outcome) : "?";
			return `Turn ${turnIndex}: glob(${pattern}) → ${fileCount} files`;
		}
		default: {
			return `Turn ${turnIndex}: ${name}(…) → ${outcome ? "done" : "?"}`;
		}
	}
}

/**
 * Find the tool result content for a given tool_use ID.
 * Tool results come back as user messages with content blocks that have
 * type "tool_result" (in Anthropic API format) or we look for adjacent user messages.
 */
function findToolResult(toolUseId: string, messages: Message[]): string | null {
	for (const msg of messages) {
		if (msg.role !== "user") continue;
		if (typeof msg.content === "string") continue;

		for (const block of msg.content) {
			// Anthropic tool_result blocks have type "tool_result"
			// We store them as user messages with content
			if (
				"type" in block &&
				block.type === "tool_use" &&
				"id" in block &&
				(block as { id: string }).id === toolUseId
			) {
				return null; // This is the call, not the result
			}
		}

		// Fall back: if there's plain text content in user message, it might be the result
		for (const block of msg.content) {
			if (block.type === "text") return block.text;
		}
	}
	return null;
}

function shortPath(filePath: string): string {
	const parts = filePath.split("/");
	if (parts.length <= 2) return filePath;
	return `…/${parts.slice(-2).join("/")}`;
}

function countLines(text: string): number {
	return text.split("\n").length;
}

function extractMatchCount(text: string): string {
	const match = /(\d+) match/i.exec(text);
	return match?.[1] ?? "?";
}

/**
 * Update the archive when a file is written or edited.
 */
export function recordFileModification(
	archive: ContextArchive,
	filePath: string,
	description: string,
): ContextArchive {
	const updated = new Map(archive.modifiedFiles);
	updated.set(filePath, description);
	return { ...archive, modifiedFiles: updated };
}

/**
 * Update the archive when an error is resolved.
 */
export function recordResolvedError(archive: ContextArchive, errorSummary: string): ContextArchive {
	return {
		...archive,
		resolvedErrors: [...archive.resolvedErrors, errorSummary],
	};
}

/**
 * Append a turn summary to the work summary.
 * If the summary would exceed maxTokens, oldest entries are dropped first.
 */
export function appendToWorkSummary(
	archive: ContextArchive,
	turnSummary: string,
	maxTokens: number,
): ContextArchive {
	const separator = "\n";
	const newSummary = archive.workSummary
		? archive.workSummary + separator + turnSummary
		: turnSummary;

	if (estimateTokens(newSummary) <= maxTokens) {
		return { ...archive, workSummary: newSummary };
	}

	// Drop oldest lines until we fit
	const lines = newSummary.split(separator);
	let trimmed = lines;
	while (trimmed.length > 1 && estimateTokens(trimmed.join(separator)) > maxTokens) {
		trimmed = trimmed.slice(1);
	}
	return { ...archive, workSummary: trimmed.join(separator) };
}

/**
 * Record a decision in the archive.
 */
export function recordDecision(archive: ContextArchive, decision: string): ContextArchive {
	return { ...archive, decisions: [...archive.decisions, decision] };
}

/**
 * Render the archive into a single markdown message string for injection.
 */
export function renderArchive(archive: ContextArchive): string {
	const sections: string[] = [];

	if (archive.workSummary) {
		sections.push(`## Work So Far\n${archive.workSummary}`);
	}

	if (archive.modifiedFiles.size > 0) {
		const files = Array.from(archive.modifiedFiles.entries())
			.map(([path, desc]) => `- ${path}: ${desc}`)
			.join("\n");
		sections.push(`## Files Modified\n${files}`);
	}

	if (archive.decisions.length > 0) {
		const decisions = archive.decisions.map((d) => `- ${d}`).join("\n");
		sections.push(`## Key Decisions\n${decisions}`);
	}

	if (archive.resolvedErrors.length > 0) {
		const errors = archive.resolvedErrors.map((e) => `- ${e}`).join("\n");
		sections.push(`## Resolved Issues\n${errors}`);
	}

	return sections.join("\n\n");
}
