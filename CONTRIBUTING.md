# Contributing to Sapling

Thanks for your interest in contributing to Sapling! This guide covers everything you need to get started.

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/sapling.git
   cd sapling
   ```
3. **Install** dependencies:
   ```bash
   bun install
   ```
4. **Link** the CLI for local development:
   ```bash
   bun link
   ```
5. **Create a branch** for your work:
   ```bash
   git checkout -b fix/description-of-change
   ```

## Branch Naming

Use descriptive branch names with a category prefix:

- `fix/` -- Bug fixes
- `feat/` -- New features
- `docs/` -- Documentation changes
- `refactor/` -- Code refactoring
- `test/` -- Test additions or fixes

## Build & Test Commands

```bash
bun test                           # Run all tests
bun test src/tools/bash.test.ts    # Run a single test file
biome check .                      # Lint + format check
biome check --write .              # Auto-fix lint + format issues
tsc --noEmit                       # Type check
bun test && biome check . && tsc --noEmit  # All quality gates
```

Always run all three quality gates before submitting a PR.

## TypeScript Conventions

Sapling is a strict TypeScript project that runs directly on Bun (no build step).

### Strict Mode

- `noUncheckedIndexedAccess` is enabled -- always handle possible `undefined` from indexing
- `noExplicitAny` is an error -- use `unknown` and narrow, or define proper types
- `useConst` is enforced -- use `const` unless reassignment is needed
- `noNonNullAssertion` is a warning -- avoid `!` postfix, check for null/undefined instead

### Minimal Runtime Dependencies

Only `chalk` (color output) and `commander` (CLI framework) are allowed as runtime dependencies. Use Bun built-in APIs:

- `Bun.spawn` for subprocesses
- `Bun.file` for file I/O
- `Bun.write` for writes
- `Bun.glob` for file discovery

External tools (`claude`, `rg`) are invoked as subprocesses via `Bun.spawn`, never as npm imports. `@anthropic-ai/sdk` is an optional dependency for the SDK backend only.

### File Organization

- All shared types and interfaces go in `src/types.ts`
- All error types go in `src/errors.ts` and must extend `SaplingError`
- Each subsystem gets its own directory under `src/` (client, tools, context, logging)
- Agent definitions (`.md` files) live in `agents/` at the repo root

### Formatting

- **Tab indentation** (enforced by Biome)
- **100 character line width** (enforced by Biome)
- Biome handles import organization automatically

## Testing Conventions

- **No mocks** unless absolutely necessary. Tests use real filesystems and real git repos.
- Create temp directories with `mkdtemp` for file I/O tests
- Use real git repos in temp directories for integration tests
- Clean up in `afterEach`
- Tests are colocated with source files: `src/tools/bash.test.ts` alongside `src/tools/bash.ts`

**Only mock when the real thing has unacceptable side effects** (LLM API calls, CC subprocess). When mocking is necessary, document WHY in a comment at the top of the test file.

Example test structure:

```typescript
import { mkdtemp, rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { afterEach, beforeEach, describe, it, expect } from "bun:test";

describe("my-feature", () => {
  let testDir: string;

  beforeEach(async () => {
    testDir = await mkdtemp(join(tmpdir(), "sapling-test-"));
  });

  afterEach(async () => {
    await rm(testDir, { recursive: true });
  });

  it("does the thing", async () => {
    // Write real files, run real code, assert real results
  });
});
```

Shared test utilities are available in `src/test-helpers.ts`:

- `createTempDir()` -- Create an isolated temp directory
- `cleanupTempDir()` -- Remove temp directories
- `createTempGitRepo()` -- Initialize a real git repo in a temp dir
- `createMockClient()` -- Returns a predictable LlmClient for loop testing

## Commit Message Style

Use concise, descriptive commit messages:

```
fix: resolve context pruning for large file reads
feat: add SDK backend with native tool_use support
docs: update CLI reference with new run flags
```

Prefix with `fix:`, `feat:`, or `docs:` when the category is clear. Plain descriptive messages are also fine.

## Pull Request Expectations

- **One concern per PR.** Keep changes focused -- a bug fix, a feature, a refactor. Not all three.
- **Tests required.** New features and bug fixes should include tests. See the testing conventions above.
- **Passing CI.** All PRs must pass CI checks (lint + typecheck + test) before merge.
- **Description.** Briefly explain what the PR does and why. Link to any relevant issues.

## Reporting Issues

Use [GitHub Issues](https://github.com/jayminwest/sapling/issues) for bug reports and feature requests. For security vulnerabilities, see [SECURITY.md](SECURITY.md).

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
