/**
 * Standardized JSON envelope helpers for Sapling CLI output.
 *
 * Success envelope: { "name": "@os-eco/sapling-cli", "version": "...", ...data }
 * Error envelope:   { "error": { "code": "...", "message": "...", "details": {...} } }
 */

import { VERSION } from "./index.ts";

export interface JsonSuccess<_T extends Record<string, unknown>> {
	name: string;
	version: string;
	[key: string]: unknown;
}

export interface JsonErrorDetail {
	code: string;
	message: string;
	details?: Record<string, unknown>;
}

export interface JsonErrorEnvelope {
	error: JsonErrorDetail;
}

/**
 * Wrap data in the standard success envelope.
 */
export function jsonOutput<T extends Record<string, unknown>>(data: T): string {
	const envelope = {
		name: "@os-eco/sapling-cli",
		version: VERSION,
		...data,
	};
	return JSON.stringify(envelope);
}

/**
 * Wrap an error in the standard error envelope.
 */
export function jsonError(
	code: string,
	message: string,
	details?: Record<string, unknown>,
): string {
	const envelope: JsonErrorEnvelope = {
		error: { code, message, ...(details ? { details } : {}) },
	};
	return JSON.stringify(envelope);
}

/**
 * Print a success envelope to stdout.
 */
export function printJson<T extends Record<string, unknown>>(data: T): void {
	console.log(jsonOutput(data));
}

/**
 * Print an error envelope to stdout (JSON mode keeps everything on stdout).
 */
export function printJsonError(
	code: string,
	message: string,
	details?: Record<string, unknown>,
): void {
	console.log(jsonError(code, message, details));
}
