// WHY MOCK: Anthropic SDK calls have real API costs and require a valid ANTHROPIC_API_KEY.
// We mock the SDK module to test response mapping and error handling without API calls.

import { describe, expect, it, mock } from "bun:test";
import { AnthropicClient } from "./anthropic.ts";
import type { LlmRequest } from "./types.ts";

const baseRequest: LlmRequest = {
	systemPrompt: "You are a helpful assistant.",
	messages: [{ role: "user", content: "Hello" }],
	tools: [],
};

function makeSdkResponse(
	overrides?: Partial<{
		content: unknown[];
		usage: unknown;
		model: string;
		stop_reason: string;
	}>,
) {
	return {
		content: overrides?.content ?? [{ type: "text", text: "Hi there!" }],
		usage: overrides?.usage ?? { input_tokens: 10, output_tokens: 5 },
		model: overrides?.model ?? "claude-sonnet-4-6",
		stop_reason: overrides?.stop_reason ?? "end_turn",
	};
}

function makeMockSdk(createFn: () => Promise<unknown>) {
	return {
		default: class MockAnthropic {
			messages = { create: createFn };
		},
	};
}

describe("AnthropicClient", () => {
	describe("estimateTokens", () => {
		const client = new AnthropicClient();

		it("estimates tokens for short text", () => {
			expect(client.estimateTokens("hello")).toBe(2);
		});

		it("returns 0 for empty string", () => {
			expect(client.estimateTokens("")).toBe(0);
		});

		it("returns 1 for 4-char string", () => {
			expect(client.estimateTokens("abcd")).toBe(1);
		});
	});

	describe("id", () => {
		it("returns anthropic-sdk", () => {
			const client = new AnthropicClient();
			expect(client.id).toBe("anthropic-sdk");
		});
	});

	describe("call", () => {
		it("maps text block response to LlmResponse correctly", async () => {
			const sdkResp = makeSdkResponse({
				content: [{ type: "text", text: "Hello!" }],
				stop_reason: "end_turn",
			});

			const client = new AnthropicClient();
			// Inject mock via dynamic import override
			mock.module("@anthropic-ai/sdk", () => makeMockSdk(() => Promise.resolve(sdkResp)));

			const result = await client.call(baseRequest);
			expect(result.content).toHaveLength(1);
			const block = result.content[0];
			expect(block?.type).toBe("text");
			if (block?.type === "text") {
				expect(block.text).toBe("Hello!");
			}
			expect(result.stopReason).toBe("end_turn");
		});

		it("maps tool_use block response correctly", async () => {
			const sdkResp = makeSdkResponse({
				content: [
					{
						type: "tool_use",
						id: "tu_abc123",
						name: "bash",
						input: { command: "ls" },
					},
				],
				stop_reason: "tool_use",
			});

			mock.module("@anthropic-ai/sdk", () => makeMockSdk(() => Promise.resolve(sdkResp)));
			const client = new AnthropicClient();

			const result = await client.call(baseRequest);
			expect(result.content).toHaveLength(1);
			const block = result.content[0];
			expect(block?.type).toBe("tool_use");
			if (block?.type === "tool_use") {
				expect(block.id).toBe("tu_abc123");
				expect(block.name).toBe("bash");
				expect(block.input).toEqual({ command: "ls" });
			}
			expect(result.stopReason).toBe("tool_use");
		});

		it("maps usage fields correctly", async () => {
			const sdkResp = makeSdkResponse({
				usage: {
					input_tokens: 100,
					output_tokens: 50,
					cache_read_input_tokens: 20,
					cache_creation_input_tokens: 5,
				},
			});

			mock.module("@anthropic-ai/sdk", () => makeMockSdk(() => Promise.resolve(sdkResp)));
			const client = new AnthropicClient();

			const result = await client.call(baseRequest);
			expect(result.usage.inputTokens).toBe(100);
			expect(result.usage.outputTokens).toBe(50);
			expect(result.usage.cacheReadTokens).toBe(20);
			expect(result.usage.cacheCreationTokens).toBe(5);
		});

		it("maps stop_reason to stopReason", async () => {
			const sdkResp = makeSdkResponse({ stop_reason: "max_tokens" });

			mock.module("@anthropic-ai/sdk", () => makeMockSdk(() => Promise.resolve(sdkResp)));
			const client = new AnthropicClient();

			const result = await client.call(baseRequest);
			expect(result.stopReason).toBe("max_tokens");
		});

		it("throws ClientError SDK_NOT_INSTALLED when SDK unavailable", async () => {
			// Simulate missing SDK by returning a module with no default export
			mock.module("@anthropic-ai/sdk", () => ({ default: null }));
			const client = new AnthropicClient();

			await expect(client.call(baseRequest)).rejects.toMatchObject({
				code: "SDK_NOT_INSTALLED",
			});
		});

		it("throws ClientError SDK_API_ERROR on API errors", async () => {
			mock.module("@anthropic-ai/sdk", () =>
				makeMockSdk(() => Promise.reject(new Error("Rate limit exceeded"))),
			);
			const client = new AnthropicClient();

			await expect(client.call(baseRequest)).rejects.toMatchObject({
				code: "SDK_API_ERROR",
			});
		});
	});
});
