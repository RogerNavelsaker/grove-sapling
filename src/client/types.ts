export interface ToolDefinition {
	name: string;
	description: string;
	input_schema: Record<string, unknown>;
}

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

export interface CcStructuredResponse {
	thinking: string;
	tool_calls?: Array<{ name: string; input: Record<string, unknown> }>;
	text_response?: string;
}
