import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import { mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DEFAULT_CONFIG, loadConfig, loadGuardConfig, validateConfig } from "./config.ts";
import { ConfigError } from "./errors.ts";

describe("validateConfig", () => {
	it("returns merged config with defaults", () => {
		const config = validateConfig({});
		expect(config.model).toBe(DEFAULT_CONFIG.model);
		expect(config.backend).toBe("cc");
		expect(config.maxTurns).toBe(200);
	});

	it("applies overrides", () => {
		const config = validateConfig({ model: "claude-opus-4-6", maxTurns: 50 });
		expect(config.model).toBe("claude-opus-4-6");
		expect(config.maxTurns).toBe(50);
	});

	it("throws ConfigError for maxTurns < 1", () => {
		expect(() => validateConfig({ maxTurns: 0 })).toThrow(ConfigError);
	});

	it("throws ConfigError for maxTurns NaN", () => {
		expect(() => validateConfig({ maxTurns: NaN })).toThrow(ConfigError);
	});

	it("throws ConfigError for maxTurns Infinity", () => {
		expect(() => validateConfig({ maxTurns: Infinity })).toThrow(ConfigError);
	});

	it("throws ConfigError for invalid backend", () => {
		expect(() => validateConfig({ backend: "invalid" as "cc" })).toThrow(ConfigError);
	});

	it("throws ConfigError for contextWindow < 1000", () => {
		expect(() => validateConfig({ contextWindow: 500 })).toThrow(ConfigError);
	});

	it("throws ConfigError for contextWindow NaN", () => {
		expect(() => validateConfig({ contextWindow: NaN })).toThrow(ConfigError);
	});

	it("throws ConfigError for contextWindow Infinity", () => {
		expect(() => validateConfig({ contextWindow: Infinity })).toThrow(ConfigError);
	});

	it("throws ConfigError when budget allocations exceed 1.0", () => {
		expect(() =>
			validateConfig({
				contextBudget: {
					windowSize: 200_000,
					allocations: {
						systemPrompt: 0.5,
						archiveSummary: 0.5,
						recentHistory: 0.5,
						currentTurn: 0.5,
						headroom: 0.5,
					},
				},
			}),
		).toThrow(ConfigError);
	});
});

describe("loadConfig", () => {
	const ENV_KEYS = [
		"SAPLING_MODEL",
		"SAPLING_BACKEND",
		"SAPLING_MAX_TURNS",
		"SAPLING_CONTEXT_WINDOW",
		"ANTHROPIC_BASE_URL",
	] as const;
	let savedEnv: Record<string, string | undefined>;

	beforeEach(() => {
		savedEnv = {};
		for (const key of ENV_KEYS) {
			savedEnv[key] = process.env[key];
			delete process.env[key];
		}
	});

	afterEach(() => {
		for (const key of ENV_KEYS) {
			if (savedEnv[key] === undefined) {
				delete process.env[key];
			} else {
				process.env[key] = savedEnv[key];
			}
		}
	});

	it("returns default config with no overrides", () => {
		const config = loadConfig();
		expect(config.model).toBe(DEFAULT_CONFIG.model);
	});

	it("applies overrides", () => {
		const config = loadConfig({ maxTurns: 10 });
		expect(config.maxTurns).toBe(10);
	});

	it("reads ANTHROPIC_BASE_URL into apiBaseUrl", () => {
		process.env.ANTHROPIC_BASE_URL = "https://api.minimax.io/anthropic";
		const config = loadConfig();
		expect(config.apiBaseUrl).toBe("https://api.minimax.io/anthropic");
	});

	it("leaves apiBaseUrl undefined when ANTHROPIC_BASE_URL is not set", () => {
		const config = loadConfig();
		expect(config.apiBaseUrl).toBeUndefined();
	});
});

describe("loadGuardConfig", () => {
	let tmpDir: string;

	beforeEach(() => {
		tmpDir = join(tmpdir(), `sapling-guards-test-${Date.now()}`);
		mkdirSync(tmpDir, { recursive: true });
	});

	it("returns null when file does not exist (standalone mode)", async () => {
		const result = await loadGuardConfig(join(tmpDir, "nonexistent.json"));
		expect(result).toBeNull();
	});

	it("parses valid guard config", async () => {
		const filePath = join(tmpDir, "guards.json");
		writeFileSync(
			filePath,
			JSON.stringify({ version: "1", rules: [{ event: "pre_tool_call", action: "allow" }] }),
		);
		const result = await loadGuardConfig(filePath);
		expect(result).not.toBeNull();
		expect(result?.rules).toHaveLength(1);
		const firstRule = result?.rules[0];
		expect(firstRule?.action).toBe("allow");
	});

	it("throws ConfigError for invalid JSON", async () => {
		const filePath = join(tmpDir, "bad.json");
		writeFileSync(filePath, "not json {{{");
		await expect(loadGuardConfig(filePath)).rejects.toThrow(ConfigError);
	});

	it("throws ConfigError when rules field is missing", async () => {
		const filePath = join(tmpDir, "no-rules.json");
		writeFileSync(filePath, JSON.stringify({ version: "1" }));
		await expect(loadGuardConfig(filePath)).rejects.toThrow(ConfigError);
	});

	it("throws ConfigError when rules is not an array", async () => {
		const filePath = join(tmpDir, "bad-rules.json");
		writeFileSync(filePath, JSON.stringify({ rules: "not-an-array" }));
		await expect(loadGuardConfig(filePath)).rejects.toThrow(ConfigError);
	});
});

describe("loadConfig CC session auto-detect", () => {
	let savedEnv: Record<string, string | undefined>;

	beforeEach(() => {
		savedEnv = {
			CLAUDECODE: process.env.CLAUDECODE,
			SAPLING_BACKEND: process.env.SAPLING_BACKEND,
		};
	});

	afterEach(() => {
		if (savedEnv.CLAUDECODE === undefined) {
			delete process.env.CLAUDECODE;
		} else {
			process.env.CLAUDECODE = savedEnv.CLAUDECODE;
		}
		if (savedEnv.SAPLING_BACKEND === undefined) {
			delete process.env.SAPLING_BACKEND;
		} else {
			process.env.SAPLING_BACKEND = savedEnv.SAPLING_BACKEND;
		}
	});

	it("auto-selects sdk when CLAUDECODE is set", () => {
		process.env.CLAUDECODE = "1";
		delete process.env.SAPLING_BACKEND;
		const config = loadConfig();
		expect(config.backend).toBe("sdk");
	});

	it("respects explicit SAPLING_BACKEND=cc even when CLAUDECODE is set", () => {
		process.env.CLAUDECODE = "1";
		process.env.SAPLING_BACKEND = "cc";
		const config = loadConfig();
		expect(config.backend).toBe("cc");
	});
});
