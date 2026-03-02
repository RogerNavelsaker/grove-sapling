/**
 * Shared test utilities for Sapling tests.
 * Import from test files: import { createTempDir, ... } from "../test-helpers.ts"
 */

import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { ContentBlock, LlmClient, LlmRequest, LlmResponse, Message } from "./types.ts";

/**
 * Create an isolated temp directory for a test.
 * Call cleanupTempDir(dir) in afterEach to remove it.
 */
export async function createTempDir(): Promise<string> {
	return mkdtemp(join(tmpdir(), "sapling-test-"));
}

/**
 * Remove a temp directory created by createTempDir.
 */
export async function cleanupTempDir(dir: string): Promise<void> {
	await rm(dir, { recursive: true, force: true });
}

/**
 * Initialize a real git repo in a temp directory.
 * Returns the path to the initialized repo.
 */
export async function createTempGitRepo(): Promise<string> {
	const dir = await createTempDir();
	const init = Bun.spawn(["git", "init", dir], {
		stdout: "pipe",
		stderr: "pipe",
	});
	const exitCode = await init.exited;
	if (exitCode !== 0) {
		const stderr = await new Response(init.stderr).text();
		throw new Error(`git init failed: ${stderr}`);
	}

	const configName = Bun.spawn(["git", "-C", dir, "config", "user.name", "test"], {
		stdout: "pipe",
		stderr: "pipe",
	});
	await configName.exited;

	const configEmail = Bun.spawn(["git", "-C", dir, "config", "user.email", "test@example.com"], {
		stdout: "pipe",
		stderr: "pipe",
	});
	await configEmail.exited;

	return dir;
}

/**
 * A predictable mock LlmClient for agent loop testing.
 * Provide a sequence of responses; they are returned in order.
 * If the sequence is exhausted, the last response is repeated.
 */
export function createMockClient(responses: LlmResponse[]): LlmClient {
	let callCount = 0;
	const calls: LlmRequest[] = [];

	return {
		id: "mock",
		call: async (request: LlmRequest): Promise<LlmResponse> => {
			calls.push(request);
			const idx = Math.min(callCount, responses.length - 1);
			callCount++;
			const response = responses[idx];
			if (!response) {
				throw new Error("Mock client has no responses configured");
			}
			return response;
		},
		estimateTokens: (text: string): number => {
			return Math.ceil(text.length / 4);
		},
		get callCount() {
			return callCount;
		},
		get calls() {
			return calls;
		},
	} as LlmClient & { callCount: number; calls: LlmRequest[] };
}

/**
 * Build a simple text response for mock client use.
 */
export function mockTextResponse(text: string): LlmResponse {
	return {
		content: [{ type: "text", text }],
		usage: { inputTokens: 100, outputTokens: 50 },
		model: "mock-model",
		stopReason: "end_turn",
	};
}

/**
 * Build a tool_use response for mock client use.
 */
export function mockToolUseResponse(
	toolName: string,
	toolInput: Record<string, unknown>,
	toolId = "tool_mock_1",
): LlmResponse {
	const block: ContentBlock = {
		type: "tool_use",
		id: toolId,
		name: toolName,
		input: toolInput,
	};
	return {
		content: [block],
		usage: { inputTokens: 100, outputTokens: 50 },
		model: "mock-model",
		stopReason: "tool_use",
	};
}

/**
 * Build a simple user message.
 */
export function userMessage(content: string): Message {
	return { role: "user", content };
}

/**
 * Build a simple assistant text message.
 */
export function assistantMessage(text: string): Message {
	return { role: "assistant", content: [{ type: "text", text }] };
}
