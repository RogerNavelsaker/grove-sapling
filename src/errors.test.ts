import { describe, expect, it } from "bun:test";
import { ClientError, ConfigError, ContextError, SaplingError, ToolError } from "./errors.ts";

describe("SaplingError", () => {
	it("sets name, message, and code", () => {
		const err = new SaplingError("something failed", "TEST_CODE");
		expect(err.name).toBe("SaplingError");
		expect(err.message).toBe("something failed");
		expect(err.code).toBe("TEST_CODE");
		expect(err instanceof Error).toBe(true);
		expect(err instanceof SaplingError).toBe(true);
	});

	it("accepts error cause via options", () => {
		const cause = new Error("root cause");
		const err = new SaplingError("wrapped", "WRAPPED", { cause });
		expect(err.cause).toBe(cause);
	});
});

describe("ClientError", () => {
	it("extends SaplingError", () => {
		const err = new ClientError("client failed", "CLIENT_ERR");
		expect(err.name).toBe("ClientError");
		expect(err instanceof SaplingError).toBe(true);
		expect(err instanceof ClientError).toBe(true);
	});
});

describe("ToolError", () => {
	it("extends SaplingError", () => {
		const err = new ToolError("tool failed", "TOOL_ERR");
		expect(err.name).toBe("ToolError");
		expect(err instanceof SaplingError).toBe(true);
		expect(err instanceof ToolError).toBe(true);
	});
});

describe("ContextError", () => {
	it("extends SaplingError", () => {
		const err = new ContextError("context error", "CTX_ERR");
		expect(err.name).toBe("ContextError");
		expect(err instanceof SaplingError).toBe(true);
		expect(err instanceof ContextError).toBe(true);
	});
});

describe("ConfigError", () => {
	it("extends SaplingError", () => {
		const err = new ConfigError("bad config", "CFG_ERR");
		expect(err.name).toBe("ConfigError");
		expect(err instanceof SaplingError).toBe(true);
		expect(err instanceof ConfigError).toBe(true);
	});
});
