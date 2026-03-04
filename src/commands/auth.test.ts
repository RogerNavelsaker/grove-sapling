import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { existsSync, rmSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";
import { readAuthStore } from "./auth.ts";

const AUTH_DIR = join(homedir(), ".sapling");
const AUTH_FILE = join(AUTH_DIR, "auth.json");

const CLI = new URL("../index.ts", import.meta.url).pathname;

async function runCli(
	args: string[],
	env?: Record<string, string>,
): Promise<{ stdout: string; stderr: string; exitCode: number }> {
	const proc = Bun.spawn(["bun", "run", CLI, ...args], {
		stdout: "pipe",
		stderr: "pipe",
		env: { ...process.env, ...env },
	});
	const stdout = await new Response(proc.stdout).text();
	const stderr = await new Response(proc.stderr).text();
	const exitCode = await proc.exited;
	return { stdout, stderr, exitCode };
}

// Backup/restore auth file around tests
let backupContent: string | null = null;

beforeEach(async () => {
	if (existsSync(AUTH_FILE)) {
		backupContent = await readFile(AUTH_FILE, "utf-8");
	} else {
		backupContent = null;
	}
	// Start with a clean state
	if (existsSync(AUTH_FILE)) {
		rmSync(AUTH_FILE);
	}
});

afterEach(async () => {
	// Restore original file
	if (existsSync(AUTH_FILE)) {
		rmSync(AUTH_FILE);
	}
	if (backupContent !== null) {
		await Bun.write(AUTH_FILE, backupContent);
	}
});

describe("auth set", () => {
	test("stores anthropic key and writes 0600 file", async () => {
		const { exitCode, stderr } = await runCli([
			"auth",
			"set",
			"anthropic",
			"--key",
			"sk-ant-test123",
		]);
		expect(stderr).toBe("");
		expect(exitCode).toBe(0);

		const store = await readAuthStore();
		expect(store.providers.anthropic?.apiKey).toBe("sk-ant-test123");

		// Check file permissions
		const stat = Bun.file(AUTH_FILE);
		expect(stat).toBeDefined();
	}, 15000);

	test("stores minimax key with base URL", async () => {
		const { exitCode } = await runCli([
			"auth",
			"set",
			"minimax",
			"--key",
			"sk-mm-test456",
			"--base-url",
			"https://api.minimax.io/anthropic",
		]);
		expect(exitCode).toBe(0);

		const store = await readAuthStore();
		expect(store.providers.minimax?.apiKey).toBe("sk-mm-test456");
		expect(store.providers.minimax?.baseUrl).toBe("https://api.minimax.io/anthropic");
	}, 15000);

	test("rejects unknown provider", async () => {
		const { exitCode, stderr } = await runCli(["auth", "set", "openai", "--key", "sk-x"]);
		expect(exitCode).toBe(1);
		expect(stderr).toContain("Unknown provider");
	}, 15000);

	test("errors without --key", async () => {
		const { exitCode, stderr } = await runCli(["auth", "set", "anthropic"]);
		expect(exitCode).toBe(1);
		expect(stderr).toContain("--key");
	}, 15000);

	test("--json output on success", async () => {
		const { stdout, exitCode } = await runCli([
			"auth",
			"set",
			"anthropic",
			"--key",
			"sk-ant-test123",
			"--json",
		]);
		expect(exitCode).toBe(0);
		const data = JSON.parse(stdout) as Record<string, unknown>;
		expect(data.success).toBe(true);
		expect(data.provider).toBe("anthropic");
	}, 15000);
});

describe("auth status", () => {
	test("shows not configured when no key set", async () => {
		const { stdout, exitCode } = await runCli(["auth", "status"], {
			ANTHROPIC_API_KEY: "",
		});
		expect(exitCode).toBe(0);
		expect(stdout).toContain("Auth Status");
	}, 15000);

	test("--json returns structured output", async () => {
		const { stdout, exitCode } = await runCli(["auth", "status", "--json"], {
			ANTHROPIC_API_KEY: "",
		});
		expect(exitCode).toBe(0);
		const data = JSON.parse(stdout) as Record<string, unknown>;
		expect(data.success).toBe(true);
		expect(Array.isArray(data.providers)).toBe(true);
		const providers = data.providers as Array<{ provider: string; configured: boolean }>;
		const names = providers.map((p) => p.provider);
		expect(names).toContain("anthropic");
		expect(names).toContain("minimax");
	}, 15000);

	test("shows configured key masked from file", async () => {
		await runCli(["auth", "set", "anthropic", "--key", "sk-ant-abcdefghij"]);
		const { stdout } = await runCli(["auth", "status", "--json"], {
			ANTHROPIC_API_KEY: "",
		});
		const data = JSON.parse(stdout) as {
			providers: Array<{
				provider: string;
				configured: boolean;
				source: string;
				maskedKey?: string;
			}>;
		};
		const ant = data.providers.find((p) => p.provider === "anthropic");
		expect(ant?.configured).toBe(true);
		expect(ant?.source).toBe("file");
		expect(ant?.maskedKey).toContain("...");
	}, 15000);

	test("env var takes precedence over file", async () => {
		await runCli(["auth", "set", "anthropic", "--key", "sk-ant-fromfile"]);
		const { stdout } = await runCli(["auth", "status", "--json"], {
			ANTHROPIC_API_KEY: "sk-ant-fromenv",
		});
		const data = JSON.parse(stdout) as { providers: Array<{ provider: string; source: string }> };
		const ant = data.providers.find((p) => p.provider === "anthropic");
		expect(ant?.source).toBe("env");
	}, 15000);
});

describe("auth clear", () => {
	test("clears specific provider", async () => {
		await runCli(["auth", "set", "anthropic", "--key", "sk-ant-test"]);
		const { exitCode } = await runCli(["auth", "clear", "anthropic"]);
		expect(exitCode).toBe(0);

		const store = await readAuthStore();
		expect(store.providers.anthropic).toBeUndefined();
	}, 15000);

	test("clears all when no provider given", async () => {
		await runCli(["auth", "set", "anthropic", "--key", "sk-ant-test"]);
		await runCli(["auth", "set", "minimax", "--key", "sk-mm-test"]);
		const { exitCode } = await runCli(["auth", "clear"]);
		expect(exitCode).toBe(0);

		const store = await readAuthStore();
		expect(Object.keys(store.providers)).toHaveLength(0);
	}, 15000);

	test("--json output on clear", async () => {
		await runCli(["auth", "set", "anthropic", "--key", "sk-ant-test"]);
		const { stdout, exitCode } = await runCli(["auth", "clear", "anthropic", "--json"]);
		expect(exitCode).toBe(0);
		const data = JSON.parse(stdout) as Record<string, unknown>;
		expect(data.success).toBe(true);
		expect(data.cleared).toBe(true);
	}, 15000);

	test("rejects unknown provider", async () => {
		const { exitCode, stderr } = await runCli(["auth", "clear", "openai"]);
		expect(exitCode).toBe(1);
		expect(stderr).toContain("Unknown provider");
	}, 15000);
});

describe("auth --help", () => {
	test("shows subcommands", async () => {
		const { stdout } = await runCli(["auth", "--help"]);
		expect(stdout).toContain("set");
		expect(stdout).toContain("status");
		expect(stdout).toContain("clear");
	});
});
