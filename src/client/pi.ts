import { ClientError } from "../errors.ts";
import { logger } from "../logging/logger.ts";
import type { ToolResultBlock } from "../types.ts";
import type { ContentBlock, LlmClient, LlmRequest, LlmResponse } from "./types.ts";

interface PiConfig {
	model?: string;
	cwd?: string;
	piPath?: string;
	timeoutMs?: number;
}

interface PiRawResponse {
	type?: string;
	thinking?: string;
	tool_calls?: Array<{ name: string; input: Record<string, unknown> }>;
	text_response?: string;
	usage?: {
		input_tokens?: number;
		output_tokens?: number;
	};
	model?: string;
}

function serializeContentBlock(block: ContentBlock | ToolResultBlock): string {
	if (block.type === "text") {
		return block.text;
	}
	if (block.type === "tool_result") {
		return block.content;
	}
	return `[Tool Call: ${block.name}(${JSON.stringify(block.input)})]`;
}

function serializeMessageContent(content: string | (ContentBlock | ToolResultBlock)[]): string {
	if (typeof content === "string") {
		return content;
	}
	return content.map(serializeContentBlock).join("\n");
}

export class PiClient implements LlmClient {
	readonly id = "pi";

	private readonly model: string | undefined;
	private readonly cwd: string;
	private readonly piPath: string;
	private readonly timeoutMs: number;

	constructor(config?: PiConfig) {
		this.model = config?.model;
		this.cwd = config?.cwd ?? process.cwd();
		this.piPath = config?.piPath ?? "pi";
		this.timeoutMs = config?.timeoutMs ?? 120_000;
	}

	estimateTokens(text: string): number {
		return Math.ceil(text.length / 4);
	}

	async call(request: LlmRequest): Promise<LlmResponse> {
		const promptLines: string[] = [];
		for (const msg of request.messages) {
			const content = serializeMessageContent(
				msg.content as string | (ContentBlock | ToolResultBlock)[],
			);
			promptLines.push(`[${msg.role === "user" ? "User" : "Assistant"}]: ${content}`);
		}
		const prompt = promptLines.join("\n");

		let systemPrompt = request.systemPrompt;
		if (request.tools.length > 0) {
			const toolDefs = request.tools
				.map(
					(t) =>
						`- **${t.name}**: ${t.description}\n  Input schema: ${JSON.stringify(t.input_schema)}`,
				)
				.join("\n");
			const toolNames = request.tools.map((t) => t.name).join(", ");
			systemPrompt = `${systemPrompt}\n\n## Available Tools\n\nYou MUST use the exact tool names listed below in your tool_calls. Do not use capitalized names, aliases, or built-in tool names.\n\n${toolDefs}\n\nIn your tool_calls JSON, the "name" field MUST exactly match one of: ${toolNames}`;
		}

		const args: string[] = [
			this.piPath,
			"-p",
			prompt,
			"--system-prompt",
			systemPrompt,
			"--output-format",
			"json",
		];

		if (request.model ?? this.model) {
			args.push("--model", (request.model ?? this.model) as string);
		}

		logger.debug("Spawning Pi subprocess", {
			pi: this.piPath,
			model: request.model ?? this.model,
			promptLength: prompt.length,
		});

		const proc = Bun.spawn(args, {
			cwd: this.cwd,
			stdin: "ignore",
			stdout: "pipe",
			stderr: "pipe",
		});

		const drainPromise = Promise.all([
			proc.exited,
			new Response(proc.stdout).text(),
			new Response(proc.stderr).text(),
		]);
		let timeoutId: ReturnType<typeof setTimeout> | undefined;
		const timeoutPromise = new Promise<never>((_, reject) => {
			timeoutId = setTimeout(() => {
				reject(new ClientError(`Pi subprocess timed out after ${this.timeoutMs}ms`, "PI_TIMEOUT"));
				proc.kill();
			}, this.timeoutMs);
		});
		const [exitCode, stdout, stderr] = await Promise.race([drainPromise, timeoutPromise]).finally(
			() => {
				clearTimeout(timeoutId);
			},
		);

		logger.debug("Pi subprocess exited", { exitCode, stdoutLength: stdout.length });

		if (exitCode !== 0) {
			throw new ClientError(`Pi subprocess failed: ${stderr}`, "PI_FAILED");
		}

		let raw: PiRawResponse;
		try {
			raw = JSON.parse(stdout) as PiRawResponse;
		} catch {
			throw new ClientError(`Pi subprocess returned invalid JSON: ${stdout}`, "PI_INVALID_JSON");
		}

		const content: ContentBlock[] = [];

		if (raw.thinking) {
			logger.debug("Pi thinking", { thinking: raw.thinking });
		}

		let stopReason: "end_turn" | "tool_use" | "max_tokens" = "end_turn";

		if (raw.tool_calls && raw.tool_calls.length > 0) {
			stopReason = "tool_use";
			for (const tc of raw.tool_calls) {
				content.push({
					type: "tool_use",
					id: crypto.randomUUID(),
					name: tc.name.toLowerCase(),
					input: tc.input,
				});
			}
		} else if (raw.text_response) {
			content.push({ type: "text", text: raw.text_response });
		}

		const usage = raw.usage ?? {};

		return {
			content,
			usage: {
				inputTokens: usage.input_tokens ?? 0,
				outputTokens: usage.output_tokens ?? 0,
			},
			model: raw.model ?? request.model ?? this.model ?? "unknown",
			stopReason,
		};
	}
}
