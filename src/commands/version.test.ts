import { describe, expect, test } from "bun:test";
import { compareSemver, getCurrentVersion } from "./version.ts";

const CLI = new URL("../index.ts", import.meta.url).pathname;

async function runCli(
	args: string[],
): Promise<{ stdout: string; stderr: string; exitCode: number }> {
	const proc = Bun.spawn(["bun", "run", CLI, ...args], {
		stdout: "pipe",
		stderr: "pipe",
	});
	const stdout = await new Response(proc.stdout).text();
	const stderr = await new Response(proc.stderr).text();
	const exitCode = await proc.exited;
	return { stdout, stderr, exitCode };
}

describe("getCurrentVersion", () => {
	test("returns a semver string", () => {
		const v = getCurrentVersion();
		expect(v).toMatch(/^\d+\.\d+\.\d+/);
	});
});

describe("version command --json", () => {
	test("outputs JSON envelope with success=true and command=version", async () => {
		const { stdout, exitCode } = await runCli(["version", "--json"]);
		expect(exitCode).toBe(0);
		const data = JSON.parse(stdout.trim()) as Record<string, unknown>;
		expect(data.success).toBe(true);
		expect(data.command).toBe("version");
		expect(typeof data.version).toBe("string");
		expect(typeof data.name).toBe("string");
		expect(typeof data.runtime).toBe("string");
		expect(typeof data.platform).toBe("string");
	});

	test("--version --json outputs JSON envelope", async () => {
		const { stdout, exitCode } = await runCli(["--version", "--json"]);
		expect(exitCode).toBe(0);
		const data = JSON.parse(stdout.trim()) as Record<string, unknown>;
		expect(data.success).toBe(true);
		expect(data.command).toBe("version");
		expect(typeof data.version).toBe("string");
	});

	test("version without --json outputs plain version string", async () => {
		const { stdout, exitCode } = await runCli(["version"]);
		expect(exitCode).toBe(0);
		expect(stdout.trim()).toMatch(/^\d+\.\d+\.\d+/);
	});
});

describe("compareSemver", () => {
	test("equal versions return 0", () => {
		expect(compareSemver("1.0.0", "1.0.0")).toBe(0);
	});

	test("a < b returns -1", () => {
		expect(compareSemver("1.0.0", "1.0.1")).toBe(-1);
		expect(compareSemver("0.9.9", "1.0.0")).toBe(-1);
		expect(compareSemver("1.1.0", "1.2.0")).toBe(-1);
	});

	test("a > b returns 1", () => {
		expect(compareSemver("1.0.1", "1.0.0")).toBe(1);
		expect(compareSemver("2.0.0", "1.9.9")).toBe(1);
	});

	test("handles missing patch segment", () => {
		expect(compareSemver("1.0", "1.0.0")).toBe(0);
		expect(compareSemver("1.1", "1.0.9")).toBe(1);
	});
});
