/**
 * ALL shared types and interfaces for Sapling.
 * Every type used across multiple modules lives here.
 */

// ─── LLM Client Types ────────────────────────────────────────────────────────

export type ContentBlock =
	| { type: "text"; text: string }
	| { type: "tool_use"; id: string; name: string; input: Record<string, unknown> };

export type Message =
	| { role: "user"; content: string | ContentBlock[] }
	| { role: "assistant"; content: ContentBlock[] };

export interface TokenUsage {
	inputTokens: number;
	outputTokens: number;
	cacheReadTokens?: number;
	cacheCreationTokens?: number;
}

export interface LlmRequest {
	systemPrompt: string;
	messages: Message[];
	tools: ToolDefinition[];
	model?: string;
	maxTokens?: number;
}

export interface LlmResponse {
	content: ContentBlock[];
	usage: TokenUsage;
	model: string;
	stopReason: "end_turn" | "tool_use" | "max_tokens";
}

export interface LlmClient {
	readonly id: string;
	call(request: LlmRequest): Promise<LlmResponse>;
	estimateTokens(text: string): number;
}

// ─── Tool Types ───────────────────────────────────────────────────────────────

export interface JsonSchema {
	type: string;
	properties?: Record<string, JsonSchema>;
	items?: JsonSchema;
	required?: string[];
	description?: string;
	enum?: unknown[];
}

export interface ToolDefinition {
	name: string;
	description: string;
	input_schema: JsonSchema;
}

export interface ToolResult {
	content: string;
	isError?: boolean;
	metadata?: {
		tokensEstimate?: number;
		filePath?: string;
		truncated?: boolean;
	};
}

export interface Tool {
	name: string;
	description: string;
	inputSchema: JsonSchema;
	execute(input: Record<string, unknown>, cwd: string): Promise<ToolResult>;
	toDefinition(): ToolDefinition;
}

export interface ToolRegistry {
	register(tool: Tool): void;
	get(name: string): Tool | undefined;
	list(): Tool[];
	toDefinitions(): ToolDefinition[];
}

// ─── Agent Loop Types ─────────────────────────────────────────────────────────

export interface LoopOptions {
	task: string;
	systemPrompt: string;
	model: string;
	maxTurns?: number;
	cwd: string;
}

export interface LoopResult {
	exitReason: "task_complete" | "max_turns" | "error" | "aborted";
	totalTurns: number;
	totalInputTokens: number;
	totalOutputTokens: number;
	error?: string;
}

// ─── Context Manager Types ────────────────────────────────────────────────────

export type MessageCategory = "system" | "task" | "archive" | "history" | "current";

export interface ScoredMessage {
	message: Message;
	score: number;
	category: MessageCategory;
	tokenCount: number;
	age: number;
	metadata: {
		filesReferenced: string[];
		isErrorContext: boolean;
		hasUnresolvedQuestion: boolean;
	};
}

export interface ContextBudget {
	windowSize: number;
	allocations: {
		systemPrompt: number;
		archiveSummary: number;
		recentHistory: number;
		currentTurn: number;
		headroom: number;
	};
}

export interface BudgetUtilization {
	systemPrompt: { used: number; budget: number };
	archiveSummary: { used: number; budget: number };
	recentHistory: { used: number; budget: number };
	currentTurn: { used: number; budget: number };
	headroom: { used: number; budget: number };
	total: { used: number; budget: number };
}

export interface ContextArchive {
	workSummary: string;
	decisions: string[];
	modifiedFiles: Map<string, string>;
	fileHashes: Map<string, string>;
	resolvedErrors: string[];
}

export interface ContextManager {
	process(messages: Message[], lastUsage: TokenUsage, currentFiles: string[]): Message[];
	getUtilization(): BudgetUtilization;
	getArchive(): ContextArchive;
}

// ─── Config Types ─────────────────────────────────────────────────────────────

export type LlmBackend = "cc" | "sdk";

export interface SaplingConfig {
	model: string;
	backend: LlmBackend;
	maxTurns: number;
	cwd: string;
	verbose: boolean;
	quiet: boolean;
	contextWindow: number;
	contextBudget: ContextBudget;
}

// ─── CLI Types ────────────────────────────────────────────────────────────────

export interface RunOptions {
	model: string;
	cwd: string;
	backend: LlmBackend;
	systemPromptFile?: string;
	maxTurns: number;
	verbose: boolean;
	json: boolean;
	quiet: boolean;
}
