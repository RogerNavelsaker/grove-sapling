/**
 * HookManager — loads guard config and provides pre/post tool call hooks.
 *
 * Currently a stub. Full implementation in sapling-8350.
 */

import type { GuardConfig } from "../types.ts";

export class HookManager {
	readonly config: GuardConfig;

	constructor(config: GuardConfig) {
		this.config = config;
	}

	/** Return true if the tool call should proceed. */
	preToolCall(_toolName: string, _input: Record<string, unknown>): boolean {
		return true;
	}

	/** Called after a tool call completes. */
	postToolCall(_toolName: string, _result: string): void {
		// stub — full implementation in sapling-8350
	}
}
