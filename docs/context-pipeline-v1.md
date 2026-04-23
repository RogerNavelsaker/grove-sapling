# Context Pipeline v1 — Design Document

**Status:** Draft
**Author:** Sapling team
**Date:** 2026-03-04
**Scope:** Complete redesign of the inter-turn context management pipeline

---

## 1. Overview & Philosophy

### What the pipeline does

The context pipeline runs between every LLM turn in Sapling's agent loop. It receives the full message history plus metadata about the current turn, and returns a curated message array and updated system prompt for the next LLM call. It is the core product — the thing that makes Sapling different from every other headless agent.

### The "minimum effective context" principle

Agent performance degrades significantly past 50-60% context window utilization. More context is not better context — it is noisier, slower, and more expensive. The pipeline's job is to curate the **minimum** set of messages that lets the agent continue its work effectively. The target is ~50% utilization under normal operation, never exceeding 60%.

This is counterintuitive. Most agent frameworks try to fill the window because "more information helps." In practice, a 200K window at 80% utilization contains 40-60K tokens of noise — old tool outputs, stale file reads, resolved error traces — that dilute the LLM's attention on the 20-30K tokens that actually matter for the current sub-task.

### Continuous curation vs. cliff-edge summarization

Every other agent framework uses the same approach: the agent runs until the context limit is hit, then all context is passed to an LLM for summarization, and the summary is given to a fresh agent to continue. This has four fundamental problems:

1. **All-or-nothing** — Performance is fine until you hit the cliff, then catastrophic information loss.
2. **Lossy bottleneck** — One LLM call decides what matters from 200K tokens. This is an impossibly hard task.
3. **Flat output** — The new agent gets a flat summary with no gradient of recency, confidence, or relevance.
4. **Reactive** — Only kicks in at the edge, when it's already too late to preserve fine-grained information.

Sapling's approach is continuous, incremental, proactive context curation. Every turn, the context is actively shaped. Old operations fade gracefully into compact summaries. The agent never hits the cliff because the cliff doesn't exist — context utilization stays flat.

### Target utilization profile

```
Turn:  1    5    10   20   50   100  150  200
       |    |    |    |    |    |    |    |
v0:   5%  15%  30%  55%  85%  HIT  ---  ---    (baseline: accumulate everything)
v1:  5%  15%  25%  40%  48%  50%  50%  50%    (target: plateau at ~50%)
```

---

## 2. The Operation Model

### What is an operation

An **operation** is a coherent unit of agent work — a sequence of turns pursuing a single sub-goal. It is the atomic unit for scoring, pruning, and archiving. The pipeline never reasons about individual messages; it reasons about operations.

Examples of operations:
- "Read and understand `src/loop.ts`" (2-3 turns: read file, maybe grep for related types)
- "Edit `src/context/prune.ts` and verify with tests" (3-5 turns: read, edit, run tests, maybe fix)
- "Investigate test failure and fix it" (4-8 turns: run tests, read error, read source, edit, re-run)
- "Explore project structure" (2-4 turns: glob, read a few files, grep for patterns)

Operations are NOT the same as individual tool calls. A single "edit-test-fix" cycle is one operation with multiple turns.

### The Turn

A **turn** is the minimum atomic unit. It consists of:
1. An assistant response (which may contain text + tool_use blocks)
2. The corresponding user message with tool_result blocks

The pipeline **never** splits below a turn. This structurally prevents the class of bugs in v0 where tool_use blocks were separated from their tool_result blocks. In v0, six known bugs stemmed from this: orphaned tool_use IDs, dangling tool_result blocks, and API errors from malformed message sequences.

```typescript
interface Turn {
	/** 0-based index within the conversation. */
	index: number;
	/** The assistant message (text + tool_use blocks). */
	assistant: Message & { role: "assistant" };
	/** The user message with tool results (may be null for the final turn). */
	toolResults: (Message & { role: "user" }) | null;
	/** Metadata extracted from this turn. */
	meta: TurnMetadata;
}

interface TurnMetadata {
	/** Tool names invoked in this turn. */
	tools: string[];
	/** File paths referenced (from tool inputs and outputs). */
	files: string[];
	/** Whether any tool result was an error. */
	hasError: boolean;
	/** Whether the assistant text contains decision language. */
	hasDecision: boolean;
	/** Estimated token count for the full turn (assistant + results). */
	tokens: number;
	/** Monotonic timestamp (Date.now()) when the turn was ingested. */
	timestamp: number;
}
```

### Operation data model

```typescript
type OperationStatus = "active" | "completed" | "compacted" | "archived";

type OperationType =
	| "explore"     // reading, grepping, globbing — gathering information
	| "mutate"      // writing, editing — changing files
	| "verify"      // running tests, linting — checking correctness
	| "investigate" // debugging — reading errors, tracing causes
	| "mixed";      // multi-type (e.g., edit + test in a tight loop)

interface Operation {
	/** Unique ID (monotonically increasing integer). */
	id: number;
	/** Current lifecycle state. */
	status: OperationStatus;
	/** Inferred operation type based on dominant tool usage. */
	type: OperationType;
	/** The turns that belong to this operation, in chronological order. */
	turns: Turn[];
	/** All file paths touched by this operation. */
	files: Set<string>;
	/** All tool names used in this operation. */
	tools: Set<string>;
	/** Whether the operation ended in success, failure, or is still in progress. */
	outcome: "success" | "failure" | "partial" | "in_progress";
	/** Key artifacts produced (file paths created/modified). */
	artifacts: string[];
	/** IDs of operations this one depends on (e.g., "investigate" depends on the "verify" that found the error). */
	dependsOn: number[];
	/** Relevance score assigned by the Evaluate stage (0.0–1.0). Updated each pipeline run. */
	score: number;
	/** Compact summary (generated when status moves to "compacted"). */
	summary: string | null;
	/** Turn index of the first turn in this operation. */
	startTurn: number;
	/** Turn index of the last turn in this operation (updated as turns are added). */
	endTurn: number;
}
```

### Operation lifecycle

```
                  new turn added
                       |
    +-------+     +--------+     +-----------+     +----------+
    | active | --> | completed | --> | compacted | --> | archived |
    +-------+     +--------+     +-----------+     +----------+
         |              |               |                |
    turns are      boundary         Compact stage     Budget stage
    accumulating   detected         generates         drops to
                                    summary           system prompt
```

- **active**: The current operation. Turns are being added. Only one operation is active at a time.
- **completed**: Boundary detected — the agent has moved on to a new sub-goal. Full turns are still retained in the message history.
- **compacted**: The Compact stage has collapsed this operation's turns into a summary. The original turns are replaced with the compact representation.
- **archived**: The Budget stage has determined this operation can't fit in the message history. Its summary is moved to the system prompt's working memory section.

### Operation type inference

The operation type is determined by the dominant tool usage pattern:

```typescript
function inferOperationType(tools: Set<string>): OperationType {
	const hasRead = tools.has("read") || tools.has("grep") || tools.has("glob");
	const hasWrite = tools.has("write") || tools.has("edit");
	const hasVerify = tools.has("bash"); // heuristic: bash is usually test/lint

	if (hasWrite && hasVerify) return "mixed";
	if (hasWrite) return "mutate";
	if (hasVerify && !hasRead) return "verify";
	if (hasRead && !hasWrite) return "explore";
	return "explore"; // default
}
```

The "investigate" type is detected heuristically: if the previous operation's outcome was "failure" and the current operation starts by reading error-related content, it's an investigation.

---

## 3. Loop <-> Pipeline Contract

### What the loop provides

The loop calls the pipeline once per turn, after tool execution, before the next LLM call. It provides:

```typescript
interface PipelineInput {
	/** Full message array (including the just-completed turn). */
	messages: Message[];
	/** The current system prompt text. */
	systemPrompt: string;
	/** Lightweight metadata about the just-completed turn. */
	turnHint: TurnHint;
	/** Token usage from the most recent LLM response. */
	usage: TokenUsage;
}

interface TurnHint {
	/** 1-based turn number. */
	turn: number;
	/** Tool names invoked this turn. */
	tools: string[];
	/** File paths from tool inputs (not outputs — those require parsing results). */
	files: string[];
	/** Whether any tool result was an error. */
	hasError: boolean;
}
```

The loop already extracts `tools` and `files` during tool dispatch (see `extractCurrentFiles` in `loop.ts`). The `TurnHint` formalizes what it already computes informally.

### What the pipeline returns

```typescript
interface PipelineOutput {
	/** Managed message array for the next LLM call. */
	messages: Message[];
	/** Updated system prompt (agent persona + working memory). */
	systemPrompt: string;
	/** Pipeline state snapshot (for RPC inspection, events, benchmarking). */
	state: PipelineState;
}

interface PipelineState {
	/** All operations (including archived). */
	operations: Operation[];
	/** The active operation's ID (or null if no active operation). */
	activeOperationId: number | null;
	/** Current context utilization (0.0–1.0). */
	utilization: number;
	/** Budget breakdown. */
	budget: BudgetUtilization;
	/** Number of operations in each status. */
	operationCounts: Record<OperationStatus, number>;
}
```

### Integration point in the loop

The pipeline replaces the current `contextManager.process()` call in `loop.ts` (line 413):

```typescript
// v0 (current):
const managed = contextManager.process(messages as Message[], response.usage, currentFiles);
messages.splice(0, messages.length, ...(managed as LoopMessage[]));

// v1 (new):
const result = pipeline.process({
	messages: messages as Message[],
	systemPrompt: options.systemPrompt,
	turnHint: { turn: totalTurns, tools: toolCalls.map(c => c.name), files: currentFiles, hasError },
	usage: response.usage,
});
messages.splice(0, messages.length, ...(result.messages as LoopMessage[]));
options.systemPrompt = result.systemPrompt; // system prompt is now dynamic
```

The key change: the system prompt is no longer static. The pipeline modifies it each turn to include the working memory section. This requires `options.systemPrompt` to become mutable, or the loop to track the "base" system prompt separately from the pipeline-managed version.

---

## 4. Pipeline Stages

```
     +--------+     +----------+     +---------+     +--------+     +--------+
     | Ingest | --> | Evaluate | --> | Compact | --> | Budget | --> | Render |
     +--------+     +----------+     +---------+     +--------+     +--------+
         |               |               |               |               |
    Group turns     Score each      Summarize low-  Enforce 50%     Build final
    into ops       operation       score ops       utilization     messages +
                                                                   system prompt
```

### 4.1 Ingest

**Purpose:** Receive raw messages and turn hints, maintain the operation registry.

**Inputs:** `PipelineInput` (messages, turnHint, usage)
**Outputs:** Updated operation registry with the new turn assigned to an operation.

#### Turn extraction

The ingest stage first converts the raw message array into `Turn` objects. It pairs each assistant message with its following user message (tool results):

```typescript
function extractTurns(messages: Message[]): Turn[] {
	const turns: Turn[] = [];
	let turnIndex = 0;

	for (let i = 0; i < messages.length; i++) {
		const msg = messages[i];
		if (msg.role !== "assistant") continue;

		const nextMsg = messages[i + 1];
		const hasResults = nextMsg && nextMsg.role === "user";

		turns.push({
			index: turnIndex++,
			assistant: msg,
			toolResults: hasResults ? nextMsg : null,
			meta: extractTurnMetadata(msg, hasResults ? nextMsg : null),
		});

		if (hasResults) i++; // skip the paired user message
	}

	return turns;
}
```

#### Boundary detection

The ingest stage determines whether the newest turn belongs to the current active operation or starts a new one. This uses a hybrid heuristic combining four signals:

| Signal | Weight | Detection |
|--------|--------|-----------|
| Tool-type transition | 0.35 | Reading -> editing, editing -> testing, testing -> reading |
| File-scope change | 0.30 | Agent moves to files with no overlap to current operation |
| Intent signal | 0.20 | Assistant text contains phrases like "now let me", "next I need to", "moving on to" |
| Temporal gap | 0.15 | >30 seconds between turns (relevant for RPC steer/followUp injection) |

A boundary is detected when the weighted score exceeds **0.5**.

```typescript
interface BoundarySignals {
	toolTypeTransition: boolean;
	fileScopeChange: boolean;
	intentSignal: boolean;
	temporalGap: boolean;
}

const BOUNDARY_WEIGHTS = {
	toolTypeTransition: 0.35,
	fileScopeChange: 0.30,
	intentSignal: 0.20,
	temporalGap: 0.15,
};

const BOUNDARY_THRESHOLD = 0.5;

function detectBoundary(
	signals: BoundarySignals,
): boolean {
	const score =
		(signals.toolTypeTransition ? BOUNDARY_WEIGHTS.toolTypeTransition : 0) +
		(signals.fileScopeChange ? BOUNDARY_WEIGHTS.fileScopeChange : 0) +
		(signals.intentSignal ? BOUNDARY_WEIGHTS.intentSignal : 0) +
		(signals.temporalGap ? BOUNDARY_WEIGHTS.temporalGap : 0);
	return score >= BOUNDARY_THRESHOLD;
}
```

**Tool-type transitions** are detected by categorizing tools into phases:

```typescript
type ToolPhase = "read" | "write" | "verify" | "search";

const TOOL_PHASES: Record<string, ToolPhase> = {
	read: "read",
	grep: "search",
	glob: "search",
	write: "write",
	edit: "write",
	bash: "verify",
};

function hasToolTransition(prevTools: Set<string>, currentTools: string[]): boolean {
	const prevPhases = new Set([...prevTools].map(t => TOOL_PHASES[t]).filter(Boolean));
	const currPhases = new Set(currentTools.map(t => TOOL_PHASES[t]).filter(Boolean));
	// Transition = current phase set has no overlap with previous
	for (const phase of currPhases) {
		if (prevPhases.has(phase)) return false;
	}
	return prevPhases.size > 0 && currPhases.size > 0;
}
```

**File-scope change** compares the current turn's files against the active operation's file set. If the Jaccard similarity is below 0.2, it's a scope change:

```typescript
function hasFileScopeChange(operationFiles: Set<string>, turnFiles: string[]): boolean {
	if (operationFiles.size === 0 || turnFiles.length === 0) return false;
	const turnSet = new Set(turnFiles);
	const intersection = [...turnSet].filter(f => operationFiles.has(f)).length;
	const union = new Set([...operationFiles, ...turnSet]).size;
	return union > 0 && intersection / union < 0.2;
}
```

**Intent signals** are detected via regex on assistant text:

```typescript
const INTENT_PATTERNS = [
	/\bnow (?:let me|I(?:'ll| will| need to| should))\b/i,
	/\bnext,?\s+I\b/i,
	/\bmoving on to\b/i,
	/\blet(?:'s| us) (?:switch|move|turn) to\b/i,
	/\bthat(?:'s| is) done[.,]?\s+/i,
	/\bwith that (?:complete|finished|done)\b/i,
];

function hasIntentSignal(assistantText: string): boolean {
	return INTENT_PATTERNS.some(p => p.test(assistantText));
}
```

#### Operation finalization

When a boundary is detected, the active operation is marked `completed` and its outcome is inferred:

```typescript
function inferOutcome(operation: Operation): Operation["outcome"] {
	const lastTurn = operation.turns[operation.turns.length - 1];
	if (!lastTurn) return "partial";

	// If the last turn had an error, the operation failed
	if (lastTurn.meta.hasError) return "failure";

	// If the operation includes writes and the last turn is a successful verify, it succeeded
	if (operation.tools.has("write") || operation.tools.has("edit")) {
		if (lastTurn.meta.tools.includes("bash") && !lastTurn.meta.hasError) return "success";
		return "partial"; // wrote but didn't verify
	}

	return "success"; // read-only operations succeed by default
}
```

### 4.2 Evaluate

**Purpose:** Score each operation for relevance to the current work.

**Inputs:** Operation registry (all operations)
**Outputs:** Updated `score` field on each operation.

The scoring formula uses five weighted signals:

| Signal | Weight | Range | Description |
|--------|--------|-------|-------------|
| Recency | 0.25 | 0.0–1.0 | Exponential decay based on turns since operation ended |
| File overlap | 0.25 | 0.0–1.0 | Jaccard similarity of operation files vs. active operation files |
| Causal dependency | 0.25 | 0.0/1.0 | Does the active operation depend on this one's output? |
| Outcome significance | 0.15 | 0.0–1.0 | Errors and decisions score higher than routine success |
| Operation type | 0.10 | 0.0–1.0 | Mutation operations score higher than exploration |

```typescript
interface EvalWeights {
	recency: number;
	fileOverlap: number;
	causalDependency: number;
	outcomeSignificance: number;
	operationType: number;
}

const EVAL_WEIGHTS: EvalWeights = {
	recency: 0.25,
	fileOverlap: 0.25,
	causalDependency: 0.25,
	outcomeSignificance: 0.15,
	operationType: 0.10,
};
```

#### Recency scoring

Uses the same exponential decay as v0, but measured in operations (not messages):

```typescript
const RECENCY_HALF_LIFE_OPS = 4; // score halves every 4 operations

function recencyScore(opsAgo: number): number {
	return Math.exp((-Math.log(2) * opsAgo) / RECENCY_HALF_LIFE_OPS);
}
```

Half-life of 4 operations means:
- 0 ops ago: 1.0
- 2 ops ago: 0.71
- 4 ops ago: 0.50
- 8 ops ago: 0.25
- 12 ops ago: 0.12

#### File overlap scoring

Jaccard similarity between the evaluated operation's file set and the active operation's file set:

```typescript
function fileOverlapScore(opFiles: Set<string>, activeFiles: Set<string>): number {
	if (opFiles.size === 0 || activeFiles.size === 0) return 0;
	const intersection = [...opFiles].filter(f => activeFiles.has(f)).length;
	const union = new Set([...opFiles, ...activeFiles]).size;
	return intersection / union;
}
```

#### Causal dependency scoring

An operation has a causal dependency on another if:
1. The active operation reads files that the other operation wrote/edited.
2. The active operation is an "investigate" type and the other operation produced the error being investigated.

```typescript
function causalDependencyScore(
	op: Operation,
	activeOp: Operation,
): number {
	// Check if active operation reads files this operation wrote
	const opArtifacts = new Set(op.artifacts);
	const activeReads = [...activeOp.files].filter(f => opArtifacts.has(f));
	if (activeReads.length > 0) return 1.0;

	// Check if active operation depends on this one explicitly
	if (activeOp.dependsOn.includes(op.id)) return 1.0;

	return 0.0;
}
```

#### Outcome significance scoring

```typescript
function outcomeSignificanceScore(op: Operation): number {
	switch (op.outcome) {
		case "failure": return 1.0;    // Errors are highly significant
		case "partial": return 0.6;    // Incomplete work may be relevant
		case "success": return 0.3;    // Routine success is less notable
		case "in_progress": return 0.8; // Active work is significant
	}
}
```

Decision content provides a bonus: if any turn in the operation contains decision language (same detection as v0), add 0.2 to the outcome score (capped at 1.0).

#### Operation type scoring

```typescript
function operationTypeScore(type: OperationType): number {
	switch (type) {
		case "mutate": return 1.0;      // Mutations have lasting effects
		case "mixed": return 0.8;       // Edit+test cycles are important
		case "verify": return 0.6;      // Test results matter
		case "investigate": return 0.7; // Debugging context is valuable
		case "explore": return 0.3;     // Exploration is often ephemeral
	}
}
```

#### Final score

```typescript
function evaluateOperation(
	op: Operation,
	activeOp: Operation | null,
	totalOps: number,
): number {
	const opsAgo = activeOp ? totalOps - 1 - op.id : 0;
	const activeFiles = activeOp?.files ?? new Set<string>();

	const score =
		EVAL_WEIGHTS.recency * recencyScore(opsAgo) +
		EVAL_WEIGHTS.fileOverlap * fileOverlapScore(op.files, activeFiles) +
		EVAL_WEIGHTS.causalDependency * (activeOp ? causalDependencyScore(op, activeOp) : 0) +
		EVAL_WEIGHTS.outcomeSignificance * outcomeSignificanceScore(op) +
		EVAL_WEIGHTS.operationType * operationTypeScore(op.type);

	return Math.min(1.0, Math.max(0.0, score));
}
```

### 4.3 Compact

**Purpose:** Replace low-scoring completed operations with compact template-based summaries. Truncate large tool outputs in higher-scoring operations.

**Inputs:** Scored operation registry
**Outputs:** Operations with status updated to "compacted" where appropriate, summaries generated.

#### Compaction thresholds

| Condition | Action |
|-----------|--------|
| `score < 0.3` and status is `completed` | Compact: replace all turns with a summary |
| `score >= 0.3` and status is `completed` | Keep full turns, but truncate large tool outputs |
| Status is `active` | Never compact — always keep full turns |

#### Compaction templates

The key improvement over v0: compaction happens at the operation level, so the summary captures **purpose**, **actions**, and **outcome** — not just "bash(cmd) -> ok".

**Template for compacted operations:**

```
[Op #{id}: {type}] {purposeSummary}
  Files: {fileList}
  Actions: {actionSummary}
  Outcome: {outcome} {outcomeDetail}
```

Example outputs:

```
[Op #3: mutate] Edited src/context/prune.ts to fix stale-read detection
  Files: src/context/prune.ts, src/context/prune.test.ts
  Actions: read(prune.ts), edit(prune.ts, lines 45-60), bash(bun test prune.test.ts)
  Outcome: success — 12 tests pass

[Op #5: investigate] Investigated "TypeError: Cannot read property 'type' of undefined"
  Files: src/loop.ts, src/context/reshape.ts
  Actions: read(loop.ts), grep("type.*undefined"), read(reshape.ts)
  Outcome: failure — error traced to orphaned tool_result at reshape.ts:45

[Op #7: explore] Explored project structure to understand benchmark system
  Files: src/bench/harness.ts, src/bench/scenarios.ts, src/types.ts
  Actions: glob(src/bench/**), read(harness.ts), read(scenarios.ts)
  Outcome: success
```

#### Purpose extraction

To generate `purposeSummary`, extract intent from the first assistant text block in the operation:

```typescript
const PURPOSE_PATTERNS: Array<{ pattern: RegExp; extract: (match: RegExpMatchArray) => string }> = [
	{ pattern: /(?:I'll|I will|Let me|I need to|I'm going to)\s+(.{10,80}?)(?:\.|$)/i,
	  extract: (m) => m[1]! },
	{ pattern: /(?:Now|Next|First),?\s+(.{10,80}?)(?:\.|$)/i,
	  extract: (m) => m[1]! },
];

function extractPurpose(operation: Operation): string {
	for (const turn of operation.turns) {
		const text = extractAssistantText(turn.assistant);
		for (const { pattern, extract } of PURPOSE_PATTERNS) {
			const match = text.match(pattern);
			if (match) return extract(match);
		}
	}
	// Fallback: describe by tools and files
	const tools = [...operation.tools].join(", ");
	const files = [...operation.files].slice(0, 3).map(shortPath).join(", ");
	return `${tools} on ${files}`;
}
```

#### Action summary generation

```typescript
function summarizeActions(operation: Operation): string {
	const actions: string[] = [];
	for (const turn of operation.turns) {
		for (const tool of turn.meta.tools) {
			const file = turn.meta.files[0];
			const arg = file ? shortPath(file) : "...";

			// Deduplicate consecutive same-tool-same-file actions
			const last = actions[actions.length - 1];
			const action = `${tool}(${arg})`;
			if (last !== action) actions.push(action);
		}
	}
	return actions.join(", ");
}
```

#### Tool output truncation (for kept operations)

Operations above the compaction threshold keep their full turns, but large tool outputs are still truncated. This reuses the v0 truncation logic but with slightly more aggressive thresholds:

| Tool output type | Max tokens | Truncation strategy |
|-----------------|------------|-------------------|
| bash stdout | 3,000 | Keep first 30 + last 15 lines |
| grep results | 1,500 | Summarize to file list + match count |
| read results | 4,000 | Keep first 60 + last 20 lines |
| glob results | 500 | Keep first 30 results |

These thresholds are lower than v0 (which used 5,000 for bash) because the 50% utilization target requires more aggressive trimming.

### 4.4 Budget

**Purpose:** Enforce the lean context target. If compacted operations still exceed budget, drop the lowest-scored operations entirely to the archive.

**Inputs:** Operation registry with compacted operations
**Outputs:** Some operations moved to "archived" status, their summaries transferred to the system prompt's working memory.

#### Budget allocation

The context window is split into three zones:

| Zone | Allocation | Purpose | Justification |
|------|-----------|---------|---------------|
| System prompt (with archive) | 25% | Agent persona + working memory | Archives need more space than v0's 15% system + 10% archive. Merging them gives more flexibility. The persona itself is ~2-4K tokens; the rest is working memory. |
| Active operations | 25% | Full turns from retained operations | This is the "working set" — the operations the agent needs to see in full. 25% of 200K = 50K tokens, which comfortably holds 5-10 operations. |
| Headroom | 50% | LLM output + safety margin | The LLM needs room to think and respond. 50% ensures the agent never feels constrained, even on complex multi-tool turns. This is the core of the "lean context" philosophy. |

```typescript
const V1_BUDGET_ALLOCATIONS = {
	systemWithArchive: 0.25, // persona + working memory
	activeOperations: 0.25,  // full turns from retained operations
	headroom: 0.50,          // LLM output + safety margin
};
```

**Why 50% headroom?** Three reasons:
1. **Output space**: The LLM generates up to 8K tokens per response. With tool calls, reasoning text, and multiple parallel tool invocations, a single turn can use 3-5K output tokens.
2. **Cache efficiency**: Anthropic's prompt caching works best when the prompt prefix is stable. With 50% headroom, the stable prefix (system prompt + early operations) stays cached even as the tail of the message array changes.
3. **Performance cliff avoidance**: Empirically, agent quality degrades past 50-60% utilization. By hard-targeting 50%, we build in margin for the occasional spike.

#### Budget enforcement algorithm

```typescript
function enforceBudget(
	operations: Operation[],
	systemPromptTokens: number,
	windowSize: number,
): { retained: Operation[]; archived: Operation[] } {
	const operationBudget = Math.floor(windowSize * V1_BUDGET_ALLOCATIONS.activeOperations);

	// Sort completed operations by score (active operation is always retained)
	const active = operations.filter(op => op.status === "active");
	const completed = operations
		.filter(op => op.status === "completed" || op.status === "compacted")
		.sort((a, b) => b.score - a.score);

	const retained: Operation[] = [...active];
	const archived: Operation[] = [];
	let usedTokens = active.reduce((sum, op) => sum + operationTokens(op), 0);

	for (const op of completed) {
		const tokens = operationTokens(op);
		if (usedTokens + tokens <= operationBudget) {
			retained.push(op);
			usedTokens += tokens;
		} else {
			archived.push(op);
		}
	}

	return { retained, archived };
}
```

#### Archive overflow

The system prompt zone (25% = 50K tokens on a 200K window) must hold both the agent persona and the working memory. If the archive grows too large:

1. Calculate available archive space: `systemBudget - personaTokens`
2. If archive exceeds available space, drop the oldest archive entries first (FIFO)
3. Log a warning when archive entries are dropped — this means the agent has been running for a very long time

### 4.5 Render

**Purpose:** Build the final message array and system prompt for the next LLM call.

**Inputs:** Retained operations (with full turns), archived operations (summaries only), base system prompt
**Outputs:** `PipelineOutput` — message array + system prompt + pipeline state

#### Message array construction

```
Messages = [task message] + [retained operations' turns, chronological]
```

The message array contains:
1. The original task message (always first, always kept)
2. Full turns from retained operations, in chronological order (by turn index)

There is no archive message in the message array. The archive lives in the system prompt. This eliminates the v0 hack of injecting synthetic `[Acknowledged]` assistant messages to maintain alternating roles.

```typescript
function renderMessages(
	taskMessage: Message,
	retainedOps: Operation[],
): Message[] {
	const messages: Message[] = [taskMessage];

	// Collect all turns from retained operations, sorted by turn index
	const allTurns = retainedOps
		.flatMap(op => op.turns)
		.sort((a, b) => a.index - b.index);

	for (const turn of allTurns) {
		messages.push(turn.assistant);
		if (turn.toolResults) {
			messages.push(turn.toolResults);
		}
	}

	return messages;
}
```

#### Compacted operation injection

When an operation has been compacted (status = "compacted"), its turns are replaced with a single synthetic exchange:

```typescript
function renderCompactedOperation(op: Operation): [Message, Message] {
	const summary = op.summary!; // guaranteed non-null when compacted

	const assistant: Message = {
		role: "assistant",
		content: [{ type: "text", text: summary }],
	};

	const ack: Message = {
		role: "user",
		content: "[continued]",
	};

	return [assistant, ack];
}
```

This maintains the alternating assistant/user pattern required by the Anthropic API while being compact.

---

## 5. System Prompt Architecture

### Composition

The system prompt is composed of three sections:

```
┌─────────────────────────────────────────────┐
│ AGENT PERSONA                               │
│ (from agents/builder.md, reviewer.md, etc.) │
│ ~2-4K tokens, stable across entire session  │
├─────────────────────────────────────────────┤
│ WORKING MEMORY                              │
│ (compact summaries of archived operations)  │
│ Grows and shrinks as operations are         │
│ archived and oldest entries are dropped     │
├─────────────────────────────────────────────┤
│ ACTIVE CONTEXT                              │
│ - Files currently modified (with status)    │
│ - Unresolved errors                         │
│ - Key decisions made                        │
│ - Current operation summary                 │
└─────────────────────────────────────────────┘
```

### Working memory section

The working memory is a chronologically ordered list of operation summaries:

```markdown
## Working Memory

### Completed Operations (oldest first)
- [Op #1: explore] Read and understood project structure. Files: src/types.ts, src/loop.ts, src/context/manager.ts. Outcome: success.
- [Op #2: mutate] Added new TurnMetadata type to src/types.ts. Outcome: success — typecheck passes.
- [Op #3: mutate] Edited src/context/prune.ts to fix stale-read detection. Files: src/context/prune.ts, src/context/prune.test.ts. Outcome: success — 12 tests pass.
- [Op #4: verify] Ran full test suite. Outcome: failure — 2 tests failing in src/context/score.test.ts.
- [Op #5: investigate] Investigated score.test.ts failures. Error: missing fileOverlap mock. Outcome: success — root cause identified.
- [Op #6: mutate] Fixed score.test.ts by adding fileOverlap mock. Outcome: success — all 520 tests pass.
```

### Active context section

The active context tracks the current state of the agent's work:

```markdown
## Active Context

**Current operation:** [Op #7: mutate] Implementing new pipeline stages
**Files modified this session:**
- src/context/ingest.ts (new)
- src/context/evaluate.ts (new)
- src/context/compact.ts (new)
- src/types.ts (edited — added Operation type)

**Unresolved errors:** None

**Key decisions:**
- Operation boundary threshold set to 0.5 (weighted score)
- Using Jaccard similarity for file overlap (not substring matching)
- Bash tool mapped to "verify" phase (covers test/lint/build use cases)
```

### Why system prompt placement is better than user messages

In v0, the archive is injected as a user message with a synthetic `[Acknowledged]` assistant message before it. This has three problems:

1. **Semantic confusion**: In Sapling's swarm context, user messages come from orchestrators and other agents. Injecting an archive as a "user message" confuses the LLM about who is speaking.

2. **Attention characteristics**: The system prompt receives consistent attention across the entire response. User messages in the middle of a conversation receive less attention (the "lost in the middle" effect). Working memory in the system prompt is more reliably attended to.

3. **Alternating-role hacks**: The Anthropic API requires alternating user/assistant messages. Injecting a user-role archive message requires a fake `[Acknowledged]` assistant message, which wastes tokens and adds fragility.

System prompt placement eliminates all three issues.

---

## 6. Template Design

### Three levels of granularity

#### Level 1: Full detail (retained operations)

Full turns are kept as-is, with tool outputs truncated per Section 4.3. No template is applied — the LLM sees the actual messages.

#### Level 2: Compact summary (compacted operations)

Used when an operation's score falls below 0.3 and it's collapsed into a single message pair.

**Template:**

```
[Operation #{id}: {type}] {purpose}
Files: {files}
Actions: {actions}
Outcome: {outcome}
{outcomeDetail}
```

**Examples by operation type:**

**explore (read-understand):**
```
[Operation #2: explore] Read and analyzed the context pipeline implementation
Files: src/context/manager.ts, src/context/score.ts, src/context/prune.ts
Actions: read(manager.ts), read(score.ts), read(prune.ts), grep("WEIGHT_")
Outcome: success
Gathered understanding of 5-stage pipeline: measure -> score -> prune -> archive -> reshape
```

**mutate (edit-test):**
```
[Operation #4: mutate] Added exponential backoff to LLM retry logic
Files: src/loop.ts, src/loop.test.ts
Actions: read(loop.ts), edit(loop.ts, lines 74-97), bash(bun test loop.test.ts)
Outcome: success — 15 tests pass including 3 new retry tests
```

**investigate (debug-fix):**
```
[Operation #6: investigate] Debugged "tool_use_id not found" API error
Files: src/loop.ts, src/context/reshape.ts, src/context/prune.ts
Actions: read(loop.ts error at line 208), grep("tool_use_id"), read(reshape.ts), read(prune.ts)
Outcome: failure — root cause identified (prune.ts drops tool_use but keeps tool_result) but not yet fixed
Key finding: pruneMessage() can split tool_use/tool_result pairs when score < 0.15
```

**verify:**
```
[Operation #8: verify] Ran full test suite and linter
Files: (project-wide)
Actions: bash(bun test), bash(bun run lint)
Outcome: success — 520 tests pass, 0 lint errors
```

#### Level 3: Archive entry (dropped to working memory)

Used in the system prompt's working memory section. One line per operation.

**Template:**

```
- [Op #{id}: {type}] {purpose}. Files: {topFiles}. Outcome: {outcome}.{keyFinding}
```

**Examples:**

```
- [Op #2: explore] Read and analyzed context pipeline. Files: manager.ts, score.ts, prune.ts. Outcome: success.
- [Op #4: mutate] Added exponential backoff to retry logic. Files: loop.ts. Outcome: success — tests pass.
- [Op #6: investigate] Debugged tool_use_id API error. Files: loop.ts, reshape.ts, prune.ts. Outcome: failure — root cause: prune splits tool pairs.
```

### Intent extraction without LLM calls

The templates above require extracting "purpose" from assistant reasoning text. This is done purely with pattern matching — no LLM calls.

**Strategy: Cascade of extraction rules**

```typescript
function extractPurpose(operation: Operation): string {
	// 1. Try to extract from first assistant text (intent statements)
	const firstText = extractAssistantText(operation.turns[0]?.assistant);
	if (firstText) {
		for (const { pattern, extract } of PURPOSE_PATTERNS) {
			const match = firstText.match(pattern);
			if (match) return capitalize(extract(match).trim());
		}
	}

	// 2. Fallback: construct from operation metadata
	return constructPurposeFromMeta(operation);
}

function constructPurposeFromMeta(op: Operation): string {
	const verb = {
		explore: "Explored",
		mutate: "Modified",
		verify: "Verified",
		investigate: "Investigated",
		mixed: "Worked on",
	}[op.type];

	const files = [...op.files].slice(0, 3).map(shortPath).join(", ");
	return `${verb} ${files}`;
}
```

The `PURPOSE_PATTERNS` array (defined in Section 4.3) covers the most common intent phrasings. When no pattern matches (roughly 20-30% of operations based on typical agent behavior), the fallback generates a serviceable purpose string from the operation's metadata.

This is deliberately conservative. A bad purpose extraction is worse than a generic one, because it gives the LLM false confidence in its understanding of past context.

---

## 7. Swarm Integration

### How Sapling operates in Overstory swarms

Sapling agents are spawned by Overstory as subprocesses. They communicate via:
- **NDJSON events** on stdout (turn_start, tool_start, tool_end, turn_end, progress, result)
- **JSON-RPC** on stdin (steer, followUp, abort, getState)
- **Lifecycle hooks** (onToolStart, onToolEnd, onSessionEnd) spawned as fire-and-forget subprocesses

The orchestrator controls the agent by:
1. Providing an initial task via the `--task` flag
2. Injecting mid-task guidance via `steer` RPC calls
3. Queuing follow-up tasks via `followUp` RPC calls
4. Reading progress via `getState` RPC calls and NDJSON events

### Messages from orchestrators in the operation model

`steer` messages are appended to the current turn's tool results (as text blocks prefixed with `[STEER]`). They are part of the current active operation — steering doesn't create a new operation because it's guidance about the current work, not a new task.

`followUp` messages are injected as standalone user messages. These **do** trigger an operation boundary because they represent a new task from the orchestrator. The ingest stage detects followUp messages by their position (standalone user message not following an assistant turn) and forces a boundary.

### How working memory helps steered agents

When the orchestrator steers an agent, the agent needs to understand its own history to respond coherently. The working memory section of the system prompt gives the steered agent:

1. **What it has done** — the list of completed operations
2. **What it knows** — files read, patterns discovered, errors encountered
3. **What it decided** — key decisions recorded in active context
4. **What state it's in** — current operation, modified files, unresolved errors

This is substantially better than v0, where a steered agent might have a pruned message history with no coherent narrative of what happened before the steer.

### Pipeline state via RPC

The `getState` RPC method should include pipeline state in its response:

```typescript
interface AgentStateSnapshot {
	status: AgentStatus;
	currentTool?: string;
	// v1 additions:
	pipeline: {
		activeOperationId: number | null;
		operationCount: number;
		contextUtilization: number;
		archiveEntryCount: number;
	};
}
```

This lets the orchestrator make informed decisions about when to steer, when to abort, and how much work the agent has completed.

---

## 8. Migration & Benchmarking

### Adapting existing benchmarks

The existing `bench/` system (harness.ts + scenarios.ts) tests the pipeline by replaying synthetic message sequences. The v1 pipeline needs the same test infrastructure, adapted to the new interface.

**Key changes:**

1. **Scenario format stays the same.** `BenchmarkScenario` defines `taskPrompt` + `messages[]` + `expectedReductionMin`. This doesn't need to change — the scenarios are synthetic conversations, and the pipeline infers operations from the message sequence.

2. **Harness changes.** `runManaged()` needs to provide `TurnHint` objects. These can be extracted from the message sequence the same way the loop does:

```typescript
function extractTurnHint(assistantMsg: Message, turnNumber: number): TurnHint {
	const tools: string[] = [];
	const files: string[] = [];
	let hasError = false;

	if (typeof assistantMsg.content !== "string") {
		for (const block of assistantMsg.content) {
			if (block.type === "tool_use") {
				tools.push(block.name);
				if (typeof block.input.file_path === "string") files.push(block.input.file_path);
			}
		}
	}

	return { turn: turnNumber, tools, files, hasError };
}
```

3. **New metrics.** In addition to the existing metrics (reduction%, context limit hits, archive coherence), v1 benchmarks should track:

| Metric | Description | Target |
|--------|-------------|--------|
| Peak utilization | Maximum context utilization across all turns | < 60% |
| Mean utilization | Average context utilization | ~50% |
| Operation count | Number of operations detected | Sanity check (should roughly match scenario structure) |
| Compaction ratio | Fraction of operations that were compacted | Scenario-dependent |
| Archive entry count | Number of operations in the archive at end | Scenario-dependent |
| Information retention | Whether the archive mentions key files/outcomes from early operations | Qualitative |

### Comparison methodology

Run both v0 and v1 on the same 14 scenarios. Compare:

1. **Token efficiency**: v1 should achieve 40-60% reduction on LONG scenarios (vs. v0's 30-50% target)
2. **Utilization stability**: v1's utilization should plateau; v0's grows monotonically
3. **Archive quality**: v1's operation-level summaries should be more informative than v0's turn-level summaries (manual inspection)
4. **Correctness**: v1 should produce zero malformed message sequences (no orphaned tool_use/tool_result blocks)

### Migration path

The migration is a clean replacement, not a gradual transition:

1. **Phase 1: Build v1 pipeline alongside v0.** New files: `src/context/v1/ingest.ts`, `evaluate.ts`, `compact.ts`, `budget.ts`, `render.ts`, `pipeline.ts`. The v0 files (`manager.ts`, `score.ts`, `prune.ts`, `archive.ts`, `reshape.ts`, `measure.ts`) stay untouched.

2. **Phase 2: Wire v1 into the loop behind a flag.** Add `--context-pipeline v1` CLI flag (default: `v0`). The loop instantiates the appropriate pipeline. Both coexist.

3. **Phase 3: Run benchmarks on both.** Compare metrics. If v1 meets targets, make it the default.

4. **Phase 4: Remove v0.** Once v1 is validated, delete the v0 files and the flag.

File structure for v1:

```
src/context/
  v1/
    pipeline.ts      # SaplingPipelineV1 — implements the 5-stage pipeline
    ingest.ts        # Turn extraction, boundary detection, operation registry
    evaluate.ts      # Operation scoring
    compact.ts       # Template-based compaction
    budget.ts        # Budget enforcement
    render.ts        # Message array + system prompt construction
    templates.ts     # Compaction and archive templates
    types.ts         # Operation, Turn, TurnMetadata, etc.
  # v0 files remain during migration:
  manager.ts
  score.ts
  prune.ts
  archive.ts
  reshape.ts
  measure.ts
```

---

## 9. Open Questions

### 1. Operation boundary accuracy

The hybrid boundary detection heuristic (Section 4.1) uses fixed weights and a threshold of 0.5. In practice:

- **False positives** (splitting one operation into two): Low cost — the pipeline just scores two small operations instead of one larger one. Compaction still works.
- **False negatives** (merging two operations into one): Higher cost — a large operation with mixed purposes will get a score that doesn't reflect either sub-goal well.

**Need:** Empirical tuning on real agent traces. Consider logging boundary decisions and scores for manual review during the Phase 2 flag period.

### 2. Token estimation accuracy

The pipeline uses the same 4-chars-per-token heuristic as v0 (`estimateTokens` in `measure.ts`). For budget enforcement at 50% utilization, estimation error has more impact than at v0's more lenient thresholds.

**Options:**
- Stick with the heuristic but add 10% safety margin to budget calculations.
- Use `tiktoken` for more accurate estimation (adds a dependency, ~5ms per call).
- Use the actual token counts from `LlmResponse.usage` to calibrate the heuristic per-session.

**Recommendation:** Start with the heuristic + 10% safety margin. If benchmarks show utilization consistently overshooting 55%, switch to `tiktoken`.

### 3. Operation dependency tracking

Section 4.2 includes causal dependency scoring, but the ingest stage doesn't currently populate `dependsOn`. Dependency detection requires understanding that "this operation reads a file that operation #3 wrote," which is straightforward for file-level dependencies but harder for logical dependencies (e.g., "this fix depends on the understanding gained in the investigation").

**Recommendation:** Implement file-level dependency tracking first. If the scoring is insufficiently nuanced, add explicit dependency detection for investigate->fix chains as a follow-up.

### 4. Steer message handling

Section 7 says steer messages don't trigger operation boundaries. But what about a steer that says "stop what you're doing and focus on X"? This is effectively a new task.

**Options:**
- Content-based detection: scan steer text for strong redirect signals ("stop", "instead", "new priority")
- Always create a boundary on steer (conservative)
- Never create a boundary on steer (current proposal)

**Recommendation:** Start with "never" (simplest). Monitor for cases where steered agents have poor context because the steer was stuffed into a wrong operation. Add content-based detection if needed.

### 5. Archive persistence across sessions

Currently, the archive is in-memory and lost when the agent process exits. For long-running swarm tasks where agents are stopped and restarted, persisting the archive to disk would let a new agent instance resume with working memory.

**Options:**
- Write archive to `.sapling/archive.json` on each pipeline run
- Include archive in the `onSessionEnd` lifecycle hook payload
- Leave it in-memory (current proposal — persistence is a separate feature)

**Recommendation:** Leave in-memory for v1. Archive persistence is a meaningful feature that deserves its own design. The system prompt architecture makes it easy to inject a loaded archive later.

### 6. LLM summarization gate

The design explicitly uses template-only summarization to establish a baseline. The templates will inevitably miss nuance that an LLM call could capture. The question is: at what point is the quality gap large enough to justify the cost?

**Criteria for adding LLM summarization:**
- Template summaries lose critical information that causes agent mistakes (measurable via benchmark regression)
- The cost of one summarization call per compacted operation is < 5% of total session cost
- Latency of summarization doesn't degrade turn-over-turn response time noticeably

**Recommendation:** Ship v1 with templates. Instrument the pipeline to log cases where the template fallback fires (no intent pattern matched). If >40% of operations use the generic fallback, that's the signal to revisit.

### 7. Multi-turn tool calls

Some tool calls span multiple messages (e.g., a bash command that prompts for input, though Sapling's bash tool doesn't support this). The current Turn model assumes one assistant + one user message per turn.

**Recommendation:** Not a concern for v1. Sapling's tools are all single-request/single-response. If multi-turn tools are added later, the Turn model needs a `continuations` field.

### 8. Budget allocation tuning

The 25/25/50 split is based on reasoning about typical agent behavior, not empirical data. The right allocation may vary by:
- Context window size (100K vs. 200K vs. 1M)
- Agent persona (builder needs more operation space than reviewer)
- Task complexity (simple tasks need less working memory)

**Recommendation:** Make the allocation configurable via `ContextBudget` (as v0 already does). The 25/25/50 split is the default. The benchmark harness should test multiple allocations to find the empirically optimal split.
