/**
 * Token counting and budget tracking for the context manager.
 *
 * Token estimation uses the 4-chars-per-token heuristic for pre-call budgeting.
 * Actual counts come from the LLM response usage field.
 */

import type { BudgetUtilization, ContentBlock, ContextBudget, Message } from "../types.ts";

/** Default context budget allocations for a 200K token window. */
export const DEFAULT_BUDGET: ContextBudget = {
	windowSize: 200_000,
	allocations: {
		systemPrompt: 0.15,
		archiveSummary: 0.1,
		recentHistory: 0.4,
		currentTurn: 0.15,
		headroom: 0.2,
	},
};

/**
 * Estimate token count for a string using the 4-chars ≈ 1 token heuristic.
 */
export function estimateTokens(text: string): number {
	return Math.ceil(text.length / 4);
}

/**
 * Estimate token count for a ContentBlock.
 */
export function estimateBlockTokens(block: ContentBlock): number {
	if (block.type === "text") {
		return estimateTokens(block.text);
	}
	// tool_use: name + JSON-serialized input
	return estimateTokens(block.name) + estimateTokens(JSON.stringify(block.input));
}

/**
 * Estimate token count for a Message.
 */
export function estimateMessageTokens(message: Message): number {
	// Role overhead (~4 tokens)
	const roleOverhead = 4;
	if (typeof message.content === "string") {
		return roleOverhead + estimateTokens(message.content);
	}
	return roleOverhead + message.content.reduce((sum, block) => sum + estimateBlockTokens(block), 0);
}

/**
 * Compute the token budget for each category given the budget config.
 */
export function computeBudgets(budget: ContextBudget): {
	systemPrompt: number;
	archiveSummary: number;
	recentHistory: number;
	currentTurn: number;
	headroom: number;
} {
	const w = budget.windowSize;
	return {
		systemPrompt: Math.floor(w * budget.allocations.systemPrompt),
		archiveSummary: Math.floor(w * budget.allocations.archiveSummary),
		recentHistory: Math.floor(w * budget.allocations.recentHistory),
		currentTurn: Math.floor(w * budget.allocations.currentTurn),
		headroom: Math.floor(w * budget.allocations.headroom),
	};
}

/**
 * Measure token usage across categorized message groups and return
 * a BudgetUtilization report.
 *
 * @param systemPromptTokens - Token count of the system prompt (estimated externally)
 * @param archiveTokens      - Token count of the rendered archive message
 * @param historyMessages    - Recent history messages
 * @param currentMessages    - Current turn messages
 * @param budget             - Budget configuration
 */
export function measureUtilization(
	systemPromptTokens: number,
	archiveTokens: number,
	historyMessages: Message[],
	currentMessages: Message[],
	budget: ContextBudget,
): BudgetUtilization {
	const budgets = computeBudgets(budget);

	const historyUsed = historyMessages.reduce((sum, m) => sum + estimateMessageTokens(m), 0);
	const currentUsed = currentMessages.reduce((sum, m) => sum + estimateMessageTokens(m), 0);

	const totalUsed = systemPromptTokens + archiveTokens + historyUsed + currentUsed;

	return {
		systemPrompt: { used: systemPromptTokens, budget: budgets.systemPrompt },
		archiveSummary: { used: archiveTokens, budget: budgets.archiveSummary },
		recentHistory: { used: historyUsed, budget: budgets.recentHistory },
		currentTurn: { used: currentUsed, budget: budgets.currentTurn },
		headroom: {
			used: Math.max(0, budget.windowSize - totalUsed),
			budget: budgets.headroom,
		},
		total: { used: totalUsed, budget: budget.windowSize },
	};
}

/**
 * Check whether the budget for a category is exceeded.
 */
export function isOverBudget(utilization: BudgetUtilization): {
	recentHistory: boolean;
	currentTurn: boolean;
	archiveSummary: boolean;
	total: boolean;
} {
	return {
		recentHistory: utilization.recentHistory.used > utilization.recentHistory.budget,
		currentTurn: utilization.currentTurn.used > utilization.currentTurn.budget,
		archiveSummary: utilization.archiveSummary.used > utilization.archiveSummary.budget,
		total: utilization.total.used > utilization.total.budget,
	};
}
