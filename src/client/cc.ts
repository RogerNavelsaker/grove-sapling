import { tmpdir } from "node:os";
import { join } from "node:path";
import { ClientError } from "../errors.ts";
import type {
	CcStructuredResponse,
	ContentBlock,
	LlmClient,
	LlmRequest,
	LlmResponse,
} from "./types.ts";

const CC_SCHEMA: Record<string, unknown> = {
	type: "object",
	properties: {
		thinking: {
			type: "string",
			description: "Reasoning about what to do next",
		},
		tool_calls: {
			type: "array",
			items: {
				type: "object",
				properties: {
					name: { type: "string" },
					input: { type: "object" },
				},
				required: ["name", "input"],
			},
		},
		text_response: {
			type: "string",
			description: "Final text when no more tools needed",
		},
	},
	required: ["thinking"],
};

interface CcConfig {
	model?: string;
	cwd?: string;
	claudePath?: string;
}

interface CcRawResponse {
	type: string;
	subtype?: string;
	result?: string;
	usage?: {
		input_tokens?: number;
		output_tokens?: number;
		cache_read_input_tokens?: number;
		cache_creation_input_tokens?: number;
	};
	model?: string;
}

function serializeContentBlock(block: ContentBlock): string {
	if (block.type === "text") {
		return block.text;
	}
	return `[Tool Call: ${block.name}(${JSON.stringify(block.input)})]`;
}

function serializeMessageContent(content: string | ContentBlock[]): string {
	if (typeof content === "string") {
		return content;
	}
	return content.map(serializeContentBlock).join("\n");
}

export class CcClient implements LlmClient {
	readonly id = "cc";

	private readonly model: string | undefined;
	private readonly cwd: string;
	private readonly claudePath: string;

	constructor(config?: CcConfig) {
		this.model = config?.model;
		this.cwd = config?.cwd ?? process.cwd();
		this.claudePath = config?.claudePath ?? "claude";
	}

	estimateTokens(text: string): number {
		return Math.ceil(text.length / 4);
	}

	async call(request: LlmRequest): Promise<LlmResponse> {
		const id = Date.now().toString(36);
		const tmpDir = tmpdir();
		const systemFile = join(tmpDir, `sapling-system-${id}.md`);
		const schemaFile = join(tmpDir, `sapling-schema-${id}.json`);

		try {
			await Bun.write(systemFile, request.systemPrompt);

			const promptLines: string[] = [];
			for (const msg of request.messages) {
				const content = serializeMessageContent(msg.content as string | ContentBlock[]);
				promptLines.push(`[${msg.role === "user" ? "User" : "Assistant"}]: ${content}`);
			}
			const prompt = promptLines.join("\n");

			const args: string[] = [
				this.claudePath,
				"-p",
				prompt,
				"--system-prompt-file",
				systemFile,
				"--tools",
				"",
				"--max-turns",
				"1",
				"--output-format",
				"json",
			];

			if (request.tools.length > 0) {
				await Bun.write(schemaFile, JSON.stringify(CC_SCHEMA));
				args.push("--json-schema", schemaFile);
			}

			if (request.model ?? this.model) {
				args.push("--model", (request.model ?? this.model) as string);
			}

			const proc = Bun.spawn(args, {
				cwd: this.cwd,
				stdout: "pipe",
				stderr: "pipe",
				env: { ...process.env },
			});

			const exitCode = await proc.exited;

			if (exitCode !== 0) {
				const stderr = await new Response(proc.stderr).text();
				throw new ClientError(`CC subprocess failed: ${stderr}`, "CC_FAILED");
			}

			const stdout = await new Response(proc.stdout).text();

			let raw: CcRawResponse;
			try {
				raw = JSON.parse(stdout) as CcRawResponse;
			} catch {
				throw new ClientError(`CC subprocess returned invalid JSON: ${stdout}`, "CC_INVALID_JSON");
			}

			if (!raw.result) {
				throw new ClientError("CC subprocess response missing result field", "CC_INVALID_RESPONSE");
			}

			let structured: CcStructuredResponse;
			try {
				structured = JSON.parse(raw.result) as CcStructuredResponse;
			} catch {
				throw new ClientError(
					`CC subprocess result is not valid JSON: ${raw.result}`,
					"CC_INVALID_JSON",
				);
			}

			const content: ContentBlock[] = [];

			if (structured.thinking) {
				content.push({ type: "text", text: structured.thinking });
			}

			let stopReason: "end_turn" | "tool_use" | "max_tokens" = "end_turn";

			if (structured.tool_calls && structured.tool_calls.length > 0) {
				stopReason = "tool_use";
				for (const tc of structured.tool_calls) {
					content.push({
						type: "tool_use",
						id: crypto.randomUUID(),
						name: tc.name,
						input: tc.input,
					});
				}
			} else if (structured.text_response) {
				content.push({ type: "text", text: structured.text_response });
			}

			const usage = raw.usage ?? {};

			return {
				content,
				usage: {
					inputTokens: usage.input_tokens ?? 0,
					outputTokens: usage.output_tokens ?? 0,
					cacheReadTokens: usage.cache_read_input_tokens,
					cacheCreationTokens: usage.cache_creation_input_tokens,
				},
				model: raw.model ?? request.model ?? this.model ?? "unknown",
				stopReason,
			};
		} finally {
			// Best-effort cleanup of temp files
			try {
				const { unlink } = await import("node:fs/promises");
				await Promise.allSettled([unlink(systemFile), unlink(schemaFile)]);
			} catch {
				// ignore cleanup errors
			}
		}
	}
}
