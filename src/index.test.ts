import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

async function runCli(
	args: string[],
): Promise<{ stdout: string; stderr: string; exitCode: number }> {
	const proc = Bun.spawn(["bun", "run", join(import.meta.dir, "index.ts"), ...args], {
		stdout: "pipe",
		stderr: "pipe",
	});
	const [stdout, stderr, exitCode] = await Promise.all([
		new Response(proc.stdout).text(),
		new Response(proc.stderr).text(),
		proc.exited,
	]);
	return { stdout, stderr, exitCode };
}

describe("--prompt-file flag", () => {
	let tmpDir: string;

	beforeEach(async () => {
		tmpDir = await mkdtemp(join(tmpdir(), "sapling-test-"));
	});

	afterEach(async () => {
		await rm(tmpDir, { recursive: true, force: true });
	});

	it("exits with error when prompt file does not exist", async () => {
		const missingPath = join(tmpDir, "nonexistent.txt");
		const { stderr, exitCode } = await runCli(["run", "--prompt-file", missingPath]);
		expect(exitCode).toBe(1);
		expect(stderr).toContain("prompt file not found");
		expect(stderr).toContain(missingPath);
	});

	it("fails with 'prompt must not be empty' when prompt file is empty", async () => {
		const promptPath = join(tmpDir, "empty.txt");
		await writeFile(promptPath, "");
		const { stderr, exitCode } = await runCli(["run", "--prompt-file", promptPath]);
		expect(exitCode).toBe(1);
		// Error is about empty prompt, not missing file — confirms file was read
		expect(stderr).toContain("prompt must not be empty");
	});
});
