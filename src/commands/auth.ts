/**
 * Auth command for sapling CLI.
 * Manages persistent API key storage in ~/.sapling/auth.json.
 */

import { chmodSync, existsSync, mkdirSync } from "node:fs";
import { readFile, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";
import type { Command } from "commander";
import { printJson } from "../json.ts";
import { colors } from "../logging/color.ts";

const AUTH_DIR = join(homedir(), ".sapling");
const AUTH_FILE = join(AUTH_DIR, "auth.json");

export interface ProviderCredentials {
	apiKey: string;
	baseUrl?: string;
}

export interface AuthStore {
	providers: Record<string, ProviderCredentials>;
}

const SUPPORTED_PROVIDERS = ["anthropic", "minimax"] as const;
type Provider = (typeof SUPPORTED_PROVIDERS)[number];

function isProvider(value: string): value is Provider {
	return (SUPPORTED_PROVIDERS as readonly string[]).includes(value);
}

export async function readAuthStore(): Promise<AuthStore> {
	if (!existsSync(AUTH_FILE)) {
		return { providers: {} };
	}
	let raw: string;
	try {
		raw = await readFile(AUTH_FILE, "utf-8");
	} catch {
		return { providers: {} };
	}
	try {
		const parsed = JSON.parse(raw) as unknown;
		if (
			typeof parsed === "object" &&
			parsed !== null &&
			"providers" in parsed &&
			typeof (parsed as Record<string, unknown>).providers === "object"
		) {
			return parsed as AuthStore;
		}
	} catch {
		// ignore parse errors
	}
	return { providers: {} };
}

async function writeAuthStore(store: AuthStore): Promise<void> {
	if (!existsSync(AUTH_DIR)) {
		mkdirSync(AUTH_DIR, { recursive: true });
	}
	await writeFile(AUTH_FILE, JSON.stringify(store, null, 2), "utf-8");
	chmodSync(AUTH_FILE, 0o600);
}

function maskKey(key: string): string {
	if (key.length <= 8) return "***";
	return `${key.slice(0, 4)}...${key.slice(-4)}`;
}

export function registerAuthCommand(program: Command): void {
	const auth = program.command("auth").description("Manage API key credentials");

	// auth set <provider>
	auth
		.command("set <provider>")
		.description("Store API key for a provider (anthropic, minimax)")
		.option("--key <apiKey>", "API key to store")
		.option("--base-url <url>", "Base URL override (required for minimax)")
		.option("--json", "Output as JSON")
		.action(async (provider: string, opts: { key?: string; baseUrl?: string; json?: boolean }) => {
			const jsonMode = opts.json ?? false;

			if (!isProvider(provider)) {
				const msg = `Unknown provider "${provider}". Supported: ${SUPPORTED_PROVIDERS.join(", ")}`;
				if (jsonMode) {
					printJson("auth set", { success: false, error: msg });
				} else {
					process.stderr.write(`Error: ${msg}\n`);
				}
				process.exitCode = 1;
				return;
			}

			const apiKey = opts.key;
			if (!apiKey) {
				const msg = "Missing required --key <apiKey>";
				if (jsonMode) {
					printJson("auth set", { success: false, error: msg });
				} else {
					process.stderr.write(`Error: ${msg}\n`);
				}
				process.exitCode = 1;
				return;
			}

			const store = await readAuthStore();
			const creds: ProviderCredentials = { apiKey };
			if (opts.baseUrl) creds.baseUrl = opts.baseUrl;
			store.providers[provider] = creds;
			await writeAuthStore(store);

			if (jsonMode) {
				printJson("auth set", { provider, configured: true });
			} else {
				process.stdout.write(`${colors.green("✓")} Stored API key for ${colors.bold(provider)}\n`);
			}
		});

	// auth status
	auth
		.command("status")
		.description("Show which providers are configured")
		.option("--json", "Output as JSON")
		.action(async (opts: { json?: boolean }) => {
			const jsonMode = opts.json ?? false;
			const store = await readAuthStore();

			const statuses = SUPPORTED_PROVIDERS.map((provider) => {
				const envKey = provider === "anthropic" ? process.env.ANTHROPIC_API_KEY : undefined;

				const fileEntry = store.providers[provider];
				let source: "env" | "file" | "none" = "none";
				let maskedKey: string | undefined;
				let baseUrl: string | undefined;

				if (envKey) {
					source = "env";
					maskedKey = maskKey(envKey);
				} else if (fileEntry) {
					source = "file";
					maskedKey = maskKey(fileEntry.apiKey);
					baseUrl = fileEntry.baseUrl;
				}

				return { provider, configured: source !== "none", source, maskedKey, baseUrl };
			});

			if (jsonMode) {
				printJson("auth status", { providers: statuses });
			} else {
				process.stdout.write(`\n${colors.bold("Auth Status")}\n\n`);
				for (const s of statuses) {
					if (s.configured) {
						const extra = s.baseUrl ? ` (${s.baseUrl})` : "";
						process.stdout.write(
							`  ${colors.green("✓")} ${colors.bold(s.provider)}: ${s.maskedKey ?? ""}${extra} [${s.source}]\n`,
						);
					} else {
						process.stdout.write(`  ${colors.dim(`- ${s.provider}: not configured`)}\n`);
					}
				}
				process.stdout.write("\n");
			}
		});

	// auth clear [provider]
	auth
		.command("clear [provider]")
		.description("Remove stored credentials (omit provider to clear all)")
		.option("--json", "Output as JSON")
		.action(async (provider: string | undefined, opts: { json?: boolean }) => {
			const jsonMode = opts.json ?? false;

			if (provider !== undefined && !isProvider(provider)) {
				const msg = `Unknown provider "${provider}". Supported: ${SUPPORTED_PROVIDERS.join(", ")}`;
				if (jsonMode) {
					printJson("auth clear", { success: false, error: msg });
				} else {
					process.stderr.write(`Error: ${msg}\n`);
				}
				process.exitCode = 1;
				return;
			}

			const store = await readAuthStore();

			if (provider) {
				const had = provider in store.providers;
				delete store.providers[provider];
				await writeAuthStore(store);
				if (jsonMode) {
					printJson("auth clear", { provider, cleared: had });
				} else {
					process.stdout.write(
						had
							? `${colors.green("✓")} Cleared credentials for ${colors.bold(provider)}\n`
							: `${colors.dim(`No credentials stored for ${provider}`)}\n`,
					);
				}
			} else {
				store.providers = {};
				await writeAuthStore(store);
				if (jsonMode) {
					printJson("auth clear", { cleared: true });
				} else {
					process.stdout.write(`${colors.green("✓")} Cleared all stored credentials\n`);
				}
			}
		});
}
