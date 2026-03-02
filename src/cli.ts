/**
 * CLI run command handler for Sapling.
 *
 * Exports runCommand() which wires together the LLM client, tool registry,
 * and context manager, then calls runLoop(). The Commander.js CLI entry point
 * (src/index.ts) imports and calls this function.
 *
 * NOTE: createClient / createToolRegistry / createContextManager are stubs
 * that will be replaced with real implementations at merge time once the
 * client, tools, and context modules land. The stub implementations provide
 * enough to run the loop with basic no-op behavior.
 */

import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { runLoop } from "./loop.ts";
import type {
	BudgetUtilization,
	ContextArchive,
	ContextManager,
	LlmClient,
	LlmRequest,
	LlmResponse,
	LoopOptions,
	Message,
	RunOptions,
	SaplingConfig,
	TokenUsage,
	Tool,
	ToolDefinition,
	ToolRegistry,
} from "./types.ts";

// ─── Stub Factories ───────────────────────────────────────────────────────────
// These will be replaced by real imports at merge time:
//   createClient     ← src/client/index.ts
//   createToolRegistry ← src/tools/index.ts
//   createContextManager ← src/context/manager.ts

/**
 * Create an LLM client for the given backend.
 * STUB: replaced at merge time by real client implementations.
 */
function createClient(config: SaplingConfig): LlmClient {
	// TODO(merge): import { createCcClient, createSdkClient } from "./client/index.ts"
	// return config.backend === "cc" ? createCcClient(config) : createSdkClient(config);
	return {
		id: `stub-${config.backend}`,
		call: async (_request: LlmRequest): Promise<LlmResponse> => {
			throw new Error(
				`LLM client stub: real client not yet wired (backend: ${config.backend}). ` +
					"Waiting for client module to land.",
			);
		},
		estimateTokens: (text: string): number => Math.ceil(text.length / 4),
	};
}

/**
 * Create a tool registry populated with all standard Sapling tools.
 * STUB: replaced at merge time by real tool implementations.
 */
function createToolRegistry(_config: SaplingConfig): ToolRegistry {
	// TODO(merge): import { buildDefaultRegistry } from "./tools/index.ts"
	// return buildDefaultRegistry(config.cwd);
	const tools = new Map<string, Tool>();
	return {
		register(tool: Tool): void {
			tools.set(tool.name, tool);
		},
		get(name: string): Tool | undefined {
			return tools.get(name);
		},
		list(): Tool[] {
			return [...tools.values()];
		},
		toDefinitions(): ToolDefinition[] {
			return [...tools.values()].map((t) => t.toDefinition());
		},
	};
}

/**
 * Create a context manager with the given budget configuration.
 * STUB: replaced at merge time by real context manager implementation.
 */
function createContextManager(_config: SaplingConfig): ContextManager {
	// TODO(merge): import { buildContextManager } from "./context/manager.ts"
	// return buildContextManager(config.contextBudget);
	return {
		process(messages: Message[], _usage: TokenUsage, _files: string[]): Message[] {
			// Pass-through: no pruning until real context manager lands
			return messages;
		},
		getUtilization(): BudgetUtilization {
			const zero = { used: 0, budget: 0 };
			return {
				systemPrompt: zero,
				archiveSummary: zero,
				recentHistory: zero,
				currentTurn: zero,
				headroom: zero,
				total: zero,
			};
		},
		getArchive(): ContextArchive {
			return {
				workSummary: "",
				decisions: [],
				modifiedFiles: new Map(),
				fileHashes: new Map(),
				resolvedErrors: [],
			};
		},
	};
}

// ─── Default System Prompt ────────────────────────────────────────────────────

const DEFAULT_SYSTEM_PROMPT = `\
You are Sapling, a coding agent. You have access to tools for reading and writing files,
running shell commands, and searching code. Work methodically: understand the task,
explore relevant code, make changes, verify results. When done, say what you accomplished.
`;

// ─── Public API ───────────────────────────────────────────────────────────────

/**
 * Execute a task using the Sapling agent loop.
 *
 * This is the handler for the `sapling run <prompt>` CLI command.
 * It sets up the LLM client, tool registry, and context manager,
 * then delegates to runLoop().
 *
 * @param prompt - The task description from the CLI
 * @param opts   - Parsed CLI options
 * @param config - Loaded and validated Sapling configuration
 * @returns The loop result (exit reason, turn count, token counts)
 */
export async function runCommand(
	prompt: string,
	opts: RunOptions,
	config: SaplingConfig,
): Promise<ReturnType<typeof runLoop>> {
	// Load custom system prompt if provided
	let systemPrompt = DEFAULT_SYSTEM_PROMPT;
	if (opts.systemPromptFile) {
		const filePath = resolve(opts.systemPromptFile);
		systemPrompt = await readFile(filePath, "utf-8");
	}

	const client = createClient(config);
	const tools = createToolRegistry(config);
	const contextManager = createContextManager(config);

	const loopOptions: LoopOptions = {
		task: prompt,
		systemPrompt,
		model: config.model,
		maxTurns: config.maxTurns,
		cwd: config.cwd,
	};

	return runLoop(client, tools, contextManager, loopOptions);
}
