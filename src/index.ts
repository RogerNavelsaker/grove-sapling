#!/usr/bin/env bun
/**
 * Sapling CLI entry point.
 * Headless coding agent with proactive context management.
 */

import { Command } from "commander";
import { runCommand } from "./cli.ts";
import { loadConfig } from "./config.ts";
import { printJson, printJsonError } from "./json.ts";
import { setColorEnabled } from "./logging/color.ts";
import { configure, logger } from "./logging/logger.ts";
import type { LlmBackend, RunOptions } from "./types.ts";

export const VERSION = "0.1.0";

const program = new Command();

program
	.name("sapling")
	.description("Headless coding agent with proactive context management")
	.version(VERSION, "-v, --version");

program
	.command("version")
	.description("Print version")
	.action(() => {
		console.log(VERSION);
	});

program
	.command("run <prompt>")
	.description("Execute a task")
	.option("--model <name>", "Model to use", "claude-sonnet-4-6")
	.option("--cwd <path>", "Working directory", process.cwd())
	.option("--backend <cc|sdk>", "LLM backend", "cc")
	.option("--system-prompt-file <path>", "Custom system prompt file")
	.option("--max-turns <n>", "Max turns", "200")
	.option("--verbose", "Log context manager decisions", false)
	.option("--json", "NDJSON event output on stdout", false)
	.option("-q, --quiet", "Suppress non-essential output", false)
	.action(async (prompt: string, opts: Record<string, unknown>) => {
		const verbose = opts.verbose as boolean;
		const quiet = opts.quiet as boolean;
		const json = opts.json as boolean;
		const systemPromptFile = opts.systemPromptFile as string | undefined;

		// Configure logger and color
		if (quiet) setColorEnabled(false);
		configure({ verbose, quiet, json });

		let config: ReturnType<typeof loadConfig>;
		try {
			config = loadConfig({
				model: opts.model as string,
				backend: opts.backend as LlmBackend,
				maxTurns: parseInt(opts.maxTurns as string, 10),
				cwd: opts.cwd as string,
				verbose,
				quiet,
			});
		} catch (err) {
			const message = err instanceof Error ? err.message : String(err);
			if (json) {
				printJsonError("CONFIG_ERROR", message);
			} else {
				logger.error(`Config error: ${message}`);
			}
			process.exit(1);
			return;
		}

		if (json) {
			printJson({
				event: "start",
				prompt,
				config: { model: config.model, backend: config.backend },
			});
		} else {
			logger.info(`Starting Sapling — model: ${config.model}, backend: ${config.backend}`);
			logger.info(`Task: ${prompt}`);
		}

		const runOpts: RunOptions = {
			model: config.model,
			cwd: config.cwd,
			backend: config.backend,
			systemPromptFile,
			maxTurns: config.maxTurns,
			verbose,
			json,
			quiet,
		};

		const result = await runCommand(prompt, runOpts, config);

		if (json) {
			printJson({
				event: "done",
				exitReason: result.exitReason,
				totalTurns: result.totalTurns,
				totalInputTokens: result.totalInputTokens,
				totalOutputTokens: result.totalOutputTokens,
				...(result.error ? { error: result.error } : {}),
			});
		} else {
			logger.info(`Done: ${result.exitReason}`, {
				turns: result.totalTurns,
				inputTokens: result.totalInputTokens,
				outputTokens: result.totalOutputTokens,
			});
		}

		process.exit(result.exitReason === "error" ? 1 : 0);
	});

program.parse(process.argv);
