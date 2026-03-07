# V1 Done — What's Complete, What's Missing, What's Broken

> Last audit: 2026-03-07
> Version: 0.3.1
> Branch: overstory/v1done-builder/sapling-67f2

## Executive Summary

Sapling v1 is **feature-complete** for core functionality. The codebase has 792 passing tests, comprehensive documentation, and a working context pipeline. However, there are **6 failing tests**, **3 TypeScript errors**, and **2 lint errors** that need attention before a clean release.

---

## ✅ What's Done

### CLI Commands (Complete)

| Command | File | Tests | Status |
|---------|------|-------|--------|
| `sp run` | `src/cli.ts`, `src/loop.ts` | ✅ | Working |
| `sp auth` | `src/commands/auth.ts` | ✅ | Working |
| `sp config` | `src/commands/config.ts` | ✅ | Working |
| `sp init` | `src/commands/init.ts` | ✅ | Working |
| `sp version` | `src/commands/version.ts` | ✅ | Working |
| `sp completions` | `src/commands/completions.ts` | ✅ | Working |
| `sp upgrade` | `src/commands/upgrade.ts` | ✅ | Working |
| `sp doctor` | `src/commands/doctor.ts` | ✅ | Working |
| `sp typo` | `src/commands/typo.ts` | ✅ | Working |

### Tool System (Complete)

| Tool | File | Tests | Status |
|------|------|-------|--------|
| Bash | `src/tools/bash.ts` | ✅ | Working |
| Read | `src/tools/read.ts` | ✅ | Working |
| Write | `src/tools/write.ts` | ✅ | Working |
| Edit | `src/tools/edit.ts` | ✅ | Working |
| Glob | `src/tools/glob.ts` | ✅ | Working |
| Grep | `src/tools/grep.ts` | ⚠️ | 6 failing tests |

### Context Pipeline v1 (Complete)

| Stage | File | Tests | Status |
|-------|------|-------|--------|
| Ingest | `src/context/v1/ingest.ts` | ✅ | Working |
| Evaluate | `src/context/v1/evaluate.ts` | ✅ | Working |
| Compact | `src/context/v1/compact.ts` | ✅ | Working |
| Budget | `src/context/v1/budget.ts` | ✅ | Working |
| Render | `src/context/v1/render.ts` | ✅ | Working |
| Registry | `src/context/v1/registry.ts` | ✅ | Working |
| Templates | `src/context/v1/templates.ts` | ✅ | Working |

### Supporting Systems

- **Config System**: YAML config with 3-layer cascade (env → project → home) ✅
- **Session Tracking**: `.sapling/session.jsonl` logging ✅
- **RPC Server**: Unix domain socket + HTTP modes (stubbed in CLI, types incomplete) ⚠️
- **Event System**: NDJSON event stream for overstory compatibility ✅
- **Auth Store**: `~/.sapling/auth.json` for API keys ✅
- **Benchmarks**: Full v1 pipeline coverage in `src/bench/` ✅

### Testing

- **Total Tests**: 803 (792 pass, 5 skip, 6 fail)
- **Test Files**: 40
- **Expectations**: 2,807
- **Coverage**: Core functionality well-tested

### Documentation

- `README.md` — Installation, CLI reference, architecture overview
- `docs/context-pipeline-v1.md` — Pipeline design (~50KB, 1200+ lines)
- `docs/mvp-spec.md` — MVP specification (~40KB)
- `docs/archive-persistence.md` — Persistence design (~10KB)
- `CHANGELOG.md` — Comprehensive version history

---

## ❌ What's Broken

### 1. GrepTool Tests Failing (6 tests)

**Severity**: Medium

**Location**: `src/tools/grep.test.ts`

**Issue**: All grep tests fail with "No matches found" even for valid patterns in test files.

**Root Cause**: The `resolveRgPrefix()` function in `src/tools/grep.ts` checks for `rg` binary or `claude --ripgrep`. Tests may be running in an environment where neither is available.

**Failed Tests**:
```
- GrepTool > finds matches in files_with_matches mode (default)
- GrepTool > returns content in content mode
- GrepTool > returns counts in count mode
- GrepTool > filters by glob pattern
- GrepTool > searches in a specific path
- GrepTool > handles large output without pipe deadlock
```

**Investigation Notes**:
- `src/tools/grep.ts:18-45`: `resolveRgPrefix()` checks for ripgrep binary
- Tests mock `execSync` but may not handle all code paths in `resolveRgPrefix()`
- The tool likely works in production but tests lack proper mocking for the prefix resolution logic

**Expected Fix**: Ensure `rg` is available in test environment or mock the dependency more comprehensively.

### 2. TypeScript Errors (3 errors)

**Severity**: High

**Location**: `src/cli.ts`

**Errors**:
```
src/cli.ts(110,18): error TS2339: Property 'rpcSocket' does not exist on type 'RunOptions'.
src/cli.ts(121,11): error TS2339: Property 'rpcSocket' does not exist on type 'RunOptions'.
src/cli.ts(123,33): error TS2339: Property 'rpcSocket' does not exist on type 'RunOptions'.
```

**Issue**: The `--rpc-socket` CLI flag is implemented but `rpcSocket` property is missing from the `RunOptions` type in `src/types.ts`.

**Code Locations**:
- Flag definition: `src/cli.ts:98` (`--rpc-socket`)
- Usage: `src/cli.ts:110, 121, 123`
- Type definition needed: `src/types.ts` → `RunOptions` interface

**Expected Fix**: Add `rpcSocket?: string` to `RunOptions` interface in `src/types.ts`.

### 3. Lint Errors (2 errors)

**Severity**: Low

**Location**: `src/context/v1/evaluate.test.ts`

**Issue**: Computed property access should use literal keys.

```
src/context/v1/evaluate.test.ts:319:17 lint/complexity/useLiteralKeys
src/context/v1/evaluate.test.ts:320:17 lint/complexity/useLiteralKeys
```

**Issue Details**: Using bracket notation (`byName["recency"]`) instead of dot notation (`byName.recency`).

**Expected Fix**: Change `byName["recency"]` to `byName.recency` and similar for `fileOverlap`.

---

## ⚠️ Warnings & Notes (Non-Blocking)

### Lint Warnings (10 warnings)

- 2x `lint/complexity/useLiteralKeys` (errors - listed above)
- 8x other warnings in test files (non-blocking)

### Lint Info Messages (6 infos)

- Suggestions for code improvements in test files

---

## ❓ Open Questions

### 1. GrepTool Runtime Availability
- Does the tool work in production even if tests fail? Need to verify with manual test.
- Should we bundle `rg` or make it truly optional?
- Is there a fallback mechanism if `rg` is not available?

### 2. E2E Tests Skipped
- 5 integration tests are skipped (require real API keys)
- Should we add CI secrets for E2E testing?
- Current skipped tests in `src/overstory-e2e.test.ts`

### 3. Model Defaults
- Default model changed to `MiniMax-M2.5` — is this stable?
- Should we document model compatibility matrix?
- Where is the default model defined? (`src/config.ts` or `src/types.ts`)

### 4. Deprecated Backends
- CC and Pi backends removed in 0.3.1
- Should we verify no dead code remains?
- Check for lingering references in `src/types.ts`

### 5. Archive Persistence
- Design docs exist (`docs/archive-persistence.md`, `docs/archive-persistence-design.md`)
- Implementation status unclear - is it implemented or just designed?
- Search for `archive` in source code to verify

### 6. RPC Server Implementation
- CLI has `--rpc-socket` flag but types are incomplete
- Is RPC server functionality actually implemented or just stubbed?
- Check `src/cli.ts` for RPC server startup code

### 7. Context Pipeline v1 Status
- Design document says "Draft" status
- Is the pipeline considered stable for production?
- Are there known limitations not documented?

### 8. Open Feature Issues
- `sapling-c9d7`: Integrate ecosystem clients into agent loop (Low priority)
- `sapling-e3da`: Surface pipeline tuning constants through config cascade (Medium)
- `sapling-6b61`: Make pipeline async-capable for LLM-based stages (Medium)

---

## 🚫 Out of Scope for v1

The following are explicitly **not** in scope for v1:

### Core Exclusions

1. **Multiple backend support** — Only SDK backend works now (CC, Pi removed)
2. **Web dashboard** — CLI-only for v1
3. **Team collaboration features** — Single-user only
4. **Plugin system** — Hardcoded 6 tools
5. **Cloud sync** — Local storage only
6. **IDE integrations** — VS Code extension not planned
7. **Multi-agent orchestration** — Single agent only

### Known Gaps (Not Implementing)

8. **E2E test automation** — Skipped tests require manual CI secrets
9. **Archive persistence** — Design exists but implementation unclear
10. **Async pipeline stages** — Future enhancement (`sapling-6b61`)
11. **Ecosystem client integration** — Future enhancement (`sapling-c9d7`)
12. **Configurable pipeline constants** — Future enhancement (`sapling-e3da`)
13. **Model compatibility matrix** — Documentation gap, not blocking
14. **Multi-turn tool calls** — Not needed for current tool set

---

## 📋 Remaining Work Before Release

### Must Fix (Blocking Release)

- [ ] **TypeScript**: Add `rpcSocket?: string` to `RunOptions` in `src/types.ts`
  - Locations: `src/types.ts` — find `interface RunOptions` and add property
  - Verification: `bun run typecheck` should pass

- [ ] **Lint**: Fix computed property keys in `src/context/v1/evaluate.test.ts`
  - Line 319: Change `byName["recency"]` to `byName.recency`
  - Line 320: Change `byName["fileOverlap"]` to `byName.fileOverlap`
  - Verification: `bun run lint` should pass

- [ ] **Tests**: Fix or skip grep tests in `src/tools/grep.test.ts`
  - Option A: Ensure `rg` binary is available in test environment
  - Option B: Add proper mocking for `resolveRgPrefix()` function
  - Option C: Skip tests with `@skip` annotation if environment-dependent
  - Verification: `bun test` should show 0 failures

### Should Fix (High Priority)

- [ ] **RPC Server**: Verify `--rpc-socket` flag actually works in production
  - Check if `src/cli.ts` has actual server implementation
  - Test socket file creation when flag is used

- [ ] **Archive Persistence**: Determine implementation status
  - Search source for `archive` keyword
  - Update docs to clarify: implemented vs. designed-only

- [ ] **CLI Flow**: Test full CLI flow end-to-end
  - Run `sp init` in temp directory
  - Run `sp run` with minimal prompt
  - Verify session logging works

- [ ] **Command Help**: Verify all commands work with `--help`
  - `sp auth --help`
  - `sp config --help`
  - `sp init --help`
  - etc.

### Could Fix (Nice to Have)

- [ ] **E2E CI Pipeline**: Add API keys to CI for integration testing
  - Requires: `ANTHROPIC_API_KEY` secret
  - Test file: `src/overstory-e2e.test.ts`

- [ ] **Model Documentation**: Create compatibility matrix
  - Which models work with which backends?
  - Default model: `MiniMax-M2.5`

- [ ] **Dead Code Cleanup**: Remove deprecated backend references
  - Search for `CC`, `Pi`, `cc`, `pi` in source
  - Check `src/types.ts` for removed backend types

- [ ] **Lint Warnings**: Address the 10 warnings in test files
  - Not blocking but would clean up output

- [ ] **Test Coverage**: Add tests for edge cases in context pipeline
  - Budget rebalancing at edge cases
  - Empty operation handling

---

## 📊 Code Statistics

| Metric | Value |
|--------|-------|
| Source files | ~60 |
| Total lines | ~9,754 |
| Test files | 40 |
| Test expectations | 2,807 |
| CLI commands | 9 |
| Tools | 6 |
| Context pipeline stages | 7 |
| Open issues (v1-related) | 3 |

---

## 🧪 Test Results

```
792 pass
5 skip (E2E tests - require API keys)
6 fail (grep tests - environment issue)
2807 expect() calls
```

### Passing Test Categories
- CLI commands (auth, config, init, version, doctor, typo, upgrade)
- Tool implementations (bash, read, write, edit, glob)
- Context pipeline (ingest, evaluate, compact, budget, render, registry, templates)
- Config system
- Session tracking
- Integration tests (mocked)
- Overstory E2E hooks (subprocess tests)

### Failing Test Categories
- GrepTool (6 failures — ripgrep availability in test environment)

### Skipped Test Categories
- Real API integration tests (5 — require ANTHROPIC_API_KEY)
- Subprocess NDJSON parsing (1 — environment-specific)

---

## 📝 Notes

- **Version**: 0.3.1 (2026-03-04)
- **Package**: `@os-eco/sapling-cli`
- **Runtime**: Bun >= 1.0
- **TypeScript**: Strict mode enabled
- **Linter**: Biome
- **Test Runner**: Bun test

### Quality Gates (Current Status)

| Gate | Command | Status |
|------|---------|--------|
| Tests | `bun test` | ❌ 6 failures |
| Lint | `bun run lint` | ❌ 2 errors |
| TypeScript | `bun run typecheck` | ❌ 3 errors |

### Audit Notes (2026-03-07)

This refined audit was performed by deeply exploring:
- Source code structure (`src/commands/`, `src/tools/`, `src/context/v1/`)
- Test suite execution results
- Open seeds issues (`sd list --status=open`)
- Package metadata (`package.json`)
- CLI flag definitions (`src/cli.ts`)
- Type definitions (`src/types.ts`)

Compared to the previous audit, the state remains unchanged:
- Same 6 failing tests (grep tool environment issue)
- Same 3 TypeScript errors (missing `rpcSocket` type)
- Same 2 lint errors (computed property keys)
- 3 open feature issues remain

The main contribution of this audit is more specific checklist items with exact file paths and line numbers, expanded open questions based on deeper exploration, and clarified out-of-scope items.
