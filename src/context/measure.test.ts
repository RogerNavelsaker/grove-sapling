/**
 * Tests for context/measure.ts — token counting and budget tracking.
 */

import { describe, expect, it } from "bun:test";
import type { ContentBlock, ContextBudget, Message } from "../types.ts";
import {
	computeBudgets,
	DEFAULT_BUDGET,
	estimateBlockTokens,
	estimateMessageTokens,
	estimateTokens,
	isOverBudget,
	measureUtilization,
} from "./measure.ts";

describe("estimateTokens", () => {
	it("returns 0 for empty string", () => {
		expect(estimateTokens("")).toBe(0);
	});

	it("estimates roughly 1 token per 4 chars", () => {
		expect(estimateTokens("abcd")).toBe(1);
		expect(estimateTokens("abcde")).toBe(2); // ceil(5/4)
		expect(estimateTokens("a".repeat(100))).toBe(25);
	});
});

describe("estimateBlockTokens", () => {
	it("estimates text block tokens", () => {
		const block: ContentBlock = { type: "text", text: "Hello world!" };
		expect(estimateBlockTokens(block)).toBe(estimateTokens("Hello world!"));
	});

	it("estimates tool_use block tokens including name and input", () => {
		const block: ContentBlock = {
			type: "tool_use",
			id: "t1",
			name: "read",
			input: { file_path: "/src/foo.ts" },
		};
		const expected =
			estimateTokens("read") + estimateTokens(JSON.stringify({ file_path: "/src/foo.ts" }));
		expect(estimateBlockTokens(block)).toBe(expected);
	});
});

describe("estimateMessageTokens", () => {
	it("adds role overhead to string content", () => {
		const msg: Message = { role: "user", content: "Hello" };
		expect(estimateMessageTokens(msg)).toBe(4 + estimateTokens("Hello"));
	});

	it("sums block tokens for array content", () => {
		const msg: Message = {
			role: "assistant",
			content: [{ type: "text", text: "Test output" }],
		};
		expect(estimateMessageTokens(msg)).toBe(4 + estimateTokens("Test output"));
	});
});

describe("computeBudgets", () => {
	it("computes correct token budgets from allocations", () => {
		const budget: ContextBudget = {
			windowSize: 100_000,
			allocations: {
				systemPrompt: 0.15,
				archiveSummary: 0.1,
				recentHistory: 0.4,
				currentTurn: 0.15,
				headroom: 0.2,
			},
		};
		const budgets = computeBudgets(budget);
		expect(budgets.systemPrompt).toBe(15_000);
		expect(budgets.archiveSummary).toBe(10_000);
		expect(budgets.recentHistory).toBe(40_000);
		expect(budgets.currentTurn).toBe(15_000);
		expect(budgets.headroom).toBe(20_000);
	});

	it("works with DEFAULT_BUDGET", () => {
		const budgets = computeBudgets(DEFAULT_BUDGET);
		expect(budgets.systemPrompt).toBe(30_000);
		expect(budgets.recentHistory).toBe(80_000);
	});
});

describe("measureUtilization", () => {
	it("returns zero utilization for empty inputs", () => {
		const util = measureUtilization(0, 0, [], [], DEFAULT_BUDGET);
		expect(util.total.used).toBe(0);
		expect(util.recentHistory.used).toBe(0);
	});

	it("measures system prompt tokens", () => {
		const util = measureUtilization(5_000, 0, [], [], DEFAULT_BUDGET);
		expect(util.systemPrompt.used).toBe(5_000);
		expect(util.systemPrompt.budget).toBe(30_000);
	});

	it("sums history message tokens", () => {
		const history: Message[] = [
			{ role: "user", content: "a".repeat(400) }, // 100 + 4 = 104 tokens
			{ role: "assistant", content: [{ type: "text", text: "b".repeat(400) }] }, // 104 tokens
		];
		const util = measureUtilization(0, 0, history, [], DEFAULT_BUDGET);
		expect(util.recentHistory.used).toBeGreaterThan(0);
		expect(util.total.used).toBe(util.recentHistory.used);
	});

	it("tracks total correctly across all categories", () => {
		const history: Message[] = [{ role: "user", content: "hello" }];
		const current: Message[] = [{ role: "assistant", content: [{ type: "text", text: "world" }] }];
		const util = measureUtilization(100, 50, history, current, DEFAULT_BUDGET);
		const histMsg = history[0];
		const currMsg = current[0];
		if (!histMsg || !currMsg) throw new Error("test setup error");
		const expected = 100 + 50 + estimateMessageTokens(histMsg) + estimateMessageTokens(currMsg);
		expect(util.total.used).toBe(expected);
	});
});

describe("isOverBudget", () => {
	it("returns false when all categories are within budget", () => {
		const util = measureUtilization(100, 100, [], [], DEFAULT_BUDGET);
		const over = isOverBudget(util);
		expect(over.recentHistory).toBe(false);
		expect(over.total).toBe(false);
	});

	it("detects when history is over budget", () => {
		// Create a 100K-token history to exceed the 80K budget
		const bigMessage: Message = {
			role: "user",
			content: "x".repeat(400_000), // ~100K tokens
		};
		const util = measureUtilization(0, 0, [bigMessage], [], DEFAULT_BUDGET);
		const over = isOverBudget(util);
		expect(over.recentHistory).toBe(true);
	});
});
