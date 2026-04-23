/**
 * Central color control for Sapling.
 * Respects NO_COLOR env var and --quiet flag.
 */

import chalk from "chalk";

/**
 * Whether color output is enabled.
 * Disabled if NO_COLOR is set or if quiet mode is active.
 */
let colorEnabled = !process.env.NO_COLOR;

export function setColorEnabled(enabled: boolean): void {
	colorEnabled = enabled;
	// chalk uses its own level; sync it
	chalk.level = enabled ? chalk.level || 3 : 0;
}

export function isColorEnabled(): boolean {
	return colorEnabled;
}

/**
 * Apply color only if color is enabled; otherwise return the string as-is.
 */
export function c(fn: (s: string) => string, text: string): string {
	return colorEnabled ? fn(text) : text;
}

// Convenience color helpers
export const colors = {
	dim: (s: string) => c(chalk.dim, s),
	bold: (s: string) => c(chalk.bold, s),
	green: (s: string) => c(chalk.green, s),
	yellow: (s: string) => c(chalk.yellow, s),
	red: (s: string) => c(chalk.red, s),
	cyan: (s: string) => c(chalk.cyan, s),
	blue: (s: string) => c(chalk.blue, s),
	magenta: (s: string) => c(chalk.magenta, s),
	gray: (s: string) => c(chalk.gray, s),
	white: (s: string) => c(chalk.white, s),
};

// Brand palette — forest green / amber / muted stone
export const brand = {
	primary: (s: string) => c((t) => chalk.rgb(76, 175, 80)(t), s),
	accent: (s: string) => c((t) => chalk.rgb(255, 183, 77)(t), s),
	muted: (s: string) => c((t) => chalk.rgb(120, 120, 110)(t), s),
	bold: (s: string) => c((t) => chalk.rgb(76, 175, 80).bold(t), s),
};

// Status icons — ecosystem Set D
export const icons = {
	open: "-",
	active: ">",
	closed: "x",
	blocked: "!",
};

/**
 * Format a success message: ✓ <message> [<id>]
 */
export function printSuccess(message: string, id?: string): string {
	const check = brand.bold("✓");
	const msg = brand.primary(message);
	const suffix = id ? ` ${brand.accent(id)}` : "";
	return `${check} ${msg}${suffix}`;
}

/**
 * Format an error message: ✗ <message>
 */
export function printError(message: string): string {
	return `${colors.red("✗")} ${colors.red(message)}`;
}

/**
 * Format a warning message: ! <message>
 */
export function printWarning(message: string): string {
	return `${colors.yellow("!")} ${colors.yellow(message)}`;
}
