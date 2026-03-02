/**
 * Structured logger for Sapling.
 * Supports human-readable output and NDJSON mode.
 */

import { colors } from "./color.ts";

export type LogLevel = "debug" | "info" | "warn" | "error";

const LOG_LEVELS: Record<LogLevel, number> = {
	debug: 0,
	info: 1,
	warn: 2,
	error: 3,
};

export interface LogEntry {
	level: LogLevel;
	message: string;
	timestamp: string;
	data?: Record<string, unknown>;
}

export interface LoggerOptions {
	/** Minimum log level to output. Default: "info" */
	level?: LogLevel;
	/** If true, output NDJSON instead of human-readable. Default: false */
	json?: boolean;
	/** If true, suppress non-essential output. Default: false */
	quiet?: boolean;
	/** If true, enable verbose/debug output. Default: false */
	verbose?: boolean;
}

export class Logger {
	private level: LogLevel;
	private json: boolean;
	private quiet: boolean;

	constructor(options: LoggerOptions = {}) {
		this.json = options.json ?? false;
		this.quiet = options.quiet ?? false;
		if (options.verbose) {
			this.level = "debug";
		} else if (options.quiet) {
			this.level = "error";
		} else {
			this.level = options.level ?? "info";
		}
	}

	private shouldLog(level: LogLevel): boolean {
		return LOG_LEVELS[level] >= LOG_LEVELS[this.level];
	}

	private format(entry: LogEntry): string {
		if (this.json) {
			return JSON.stringify(entry);
		}

		const ts = colors.dim(entry.timestamp);
		const levelLabel = this.formatLevel(entry.level);
		const msg = entry.message;
		const data =
			entry.data && Object.keys(entry.data).length > 0
				? ` ${colors.dim(JSON.stringify(entry.data))}`
				: "";

		return `${ts} ${levelLabel} ${msg}${data}`;
	}

	private formatLevel(level: LogLevel): string {
		switch (level) {
			case "debug":
				return colors.gray("DBG");
			case "info":
				return colors.cyan("INF");
			case "warn":
				return colors.yellow("WRN");
			case "error":
				return colors.red("ERR");
		}
	}

	private write(entry: LogEntry): void {
		const line = this.format(entry);
		if (entry.level === "error") {
			process.stderr.write(`${line}\n`);
		} else {
			process.stdout.write(`${line}\n`);
		}
	}

	private log(level: LogLevel, message: string, data?: Record<string, unknown>): void {
		if (!this.shouldLog(level)) return;
		if (this.quiet && level !== "error") return;

		const entry: LogEntry = {
			level,
			message,
			timestamp: new Date().toISOString(),
			...(data ? { data } : {}),
		};
		this.write(entry);
	}

	debug(message: string, data?: Record<string, unknown>): void {
		this.log("debug", message, data);
	}

	info(message: string, data?: Record<string, unknown>): void {
		this.log("info", message, data);
	}

	warn(message: string, data?: Record<string, unknown>): void {
		this.log("warn", message, data);
	}

	error(message: string, data?: Record<string, unknown>): void {
		this.log("error", message, data);
	}
}

/** Default global logger instance. Reconfigure with configure(). */
let globalLogger = new Logger();

export function configure(options: LoggerOptions): void {
	globalLogger = new Logger(options);
}

export const logger = {
	debug: (msg: string, data?: Record<string, unknown>) => globalLogger.debug(msg, data),
	info: (msg: string, data?: Record<string, unknown>) => globalLogger.info(msg, data),
	warn: (msg: string, data?: Record<string, unknown>) => globalLogger.warn(msg, data),
	error: (msg: string, data?: Record<string, unknown>) => globalLogger.error(msg, data),
};
