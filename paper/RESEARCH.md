# Intent-Directed Compilation: From 9-Channel Semantics to Better Machine Code

## The Problem

Traditional compilers optimize **syntax** — they see instructions, not intent. They don't know:
- Whether a constraint check is safety-critical or advisory
- Whether speed matters more than precision for THIS specific check
- Whether the receiver can tolerate approximation
- Whether negative knowledge (absence of violation) is sufficient

Our compilation chain currently produces the same machine code regardless of what the constraint MEANS. A DO-178C flight control check gets the same VPCMPD as a UI color validation.

**This is the gap.** The 9-channel model can close it.

---

## The Current Compilation Chain

```
CDCL Solver → CDCLTrace (decisions, propagations, conflicts, backtracks)
    ↓
LLVMEmitter → LLVM IR text (.ll format)
    ↓                              ↓
LLVM opt     OR   x86-64 Emitter (Oracle1's direct codegen)
    ↓                              ↓
AVX-512 object code           Raw machine code (mmap, PROT_EXEC)
    ↓                              ↓
Constraint check:              Three variants:
  (<16 x i32>, <16 x i32>)       1. build_check_constraints() — 16-lane bounds check
  → i1 (pass/fail)                2. build_bloom_check() — probabilistic membership
                                  3. build_batch_check_all() — batch 10M+ constraints
```

### What Oracle1 Built (x86-64 Emitter)

The direct emitter produces raw AVX-512 machine code:
- **VMOVDQU64** — load 512-bit vectors (16 × INT32)
- **VPCMPD** — parallel comparison (SGE/SLE/ULT/EQ/NE)
- **KPANDW/KPORW** — mask combination (AND/OR across lanes)
- **VPTESTMD** — test mask register (any/all pass)
- **Bloom filter** — XOR fingerprint + POPCNT threshold (12 bits)

### What's Missing: Intent Metadata

The CDCLTrace carries NO semantic information about what the constraints MEAN. It's purely structural:
- `Decision { level, literal, reason }`
- `Propagation { literal, antecedent, level }`
- `Conflict { level, clause, analysis }`
- `Backtrack { target, learnt }`

The x86-64 emitter generates code from this trace but has ZERO knowledge of:
- Which constraints are safety-critical (C9)
- What precision is needed (C4 × tolerance)
- Whether negative knowledge suffices (C3)
- What the receiver's draft is (draft_check)

---

## Intent-Directed Compilation: The Architecture

### Level 1: Annotated Bytecode

Current bytecode (flux-vm, 50 opcodes):
```
CMP r1, r2, GE    ; Compare r1 >= r2
AND mask1, mask2  ; Combine results
TEST mask         ; Check if any fail
RET               ; Return pass/fail
```

Proposed intent-annotated bytecode:
```
CMP r1, r2, GE @intent(C1:0.9, C4:0.6, C9:0.95, tol:0.05)
    ; ^ Safety-critical boundary check: high stakes, tight tolerance
AND mask1, mask2
TEST mask
RET @verify(dual_redundancy)  ; Emit twice, compare results
```

The intent metadata doesn't change the semantics but tells the code generator HOW to emit:
- **C9 > 0.8, tol < 0.1** → Dual-redundant AVX-512 (compute twice, XOR masks, assert zero)
- **C9 < 0.3, tol > 0.5** → INT8 fast path (single VPCMPD, no redundancy)
- **C4 > 0.8** → Add telemetry (store mask register to trace buffer)
- **C3 (Process) high** → Add temporal checks (compare against previous state)

### Level 2: Negative Knowledge Optimization

**The immune system approach:** Instead of checking all possible violations, model what's SAFE and check for deviations.

Current approach:
```
for each constraint:
    check(lower <= value <= upper)  // O(n) checks
```

Negative knowledge approach:
```
compute safe_envelope(all_constraints)  // O(n) once
check(value inside safe_envelope)       // O(1) per value
```

**Concrete optimization:** If 8 constraints define a convex region, one AVX-512 VPCMPD can check all 8 simultaneously. But if we know the safe region is CONTIGUOUS (negative knowledge: "we know where violations AREN'T"), we can:
1. Pre-compute the Bloom filter of the safe region
2. Check membership probabilistically first (1 instruction)
3. Only fall back to exact check on Bloom miss

This is exactly what `build_bloom_check()` does — but it's applied uniformly. With intent metadata:
- **Safety-critical (C9 high):** Skip Bloom, go straight to exact check
- **Advisory (C9 low):** Bloom-only, no exact check needed
- **Mixed:** Bloom fast-path with exact fallback

### Level 3: Tolerance-Directed Precision

Current code uses INT32 for everything. But the 9-channel tolerance says HOW MUCH deviation is acceptable:

| Tolerance | Data Type | Throughput | Use Case |
|-----------|-----------|------------|----------|
| > 0.5 | INT8 (8 constraints/byte) | 341B c/s | UI, logging, telemetry |
| 0.2–0.5 | INT16 (4 constraints/short) | ~200B c/s | Work validation, scheduling |
| 0.05–0.2 | INT32 (1 constraint/int) | ~85B c/s | Technical specs, API contracts |
| < 0.05 | FP64 + dual redundancy | ~20B c/s | DO-178C, ISO 26262, medical |

**The key insight:** Most constraints in a real system are advisory (tolerance > 0.5). Only a tiny fraction need INT32 or better. Intent-directed compilation can:
1. Classify constraints by tolerance from the 9-channel profile
2. Emit INT8 fast-path for 85% of constraints
3. Emit INT32 exact-path for 14% of constraints
4. Emit dual-redundant FP64 for 1% of constraints
5. **Overall throughput: 5-10x improvement** because most work goes through INT8

### Level 4: Draft-Aware Instruction Scheduling

Draft = how much shared context the constraint requires. In machine code terms:

**Low draft (0.0-0.3):** Constraint is self-contained. Can be:
- Hoisted out of loops (invariant)
- Computed at compile time (constant folding)
- Cached across invocations

**Medium draft (0.3-0.7):** Constraint depends on some context. Emit:
- Normal check in the hot path
- Prefetch context before the check

**High draft (0.7-1.0):** Constraint depends on deep context. Emit:
- Conservative ordering (no speculative execution past the check)
- Cache-line alignment for the constraint data
- Branch prediction hints (likely/unlikely)

### Level 5: The Fitting Selector for Code Generation

The hydraulic fitting metaphor maps directly to code generation strategies:

```
Fitting          │ Precision    │ Redundancy │ Throughput │ Code
─────────────────┼──────────────┼────────────┼────────────┼──────────────
HoseClamp        │ INT8         │ None       │ 341B c/s   │ Single VPCMPD
(50 PSI)         │              │            │            │ + SETcc
─────────────────┼──────────────┼────────────┼────────────┼──────────────
IndustrialFitting│ INT16        │ None       │ ~200B c/s  │ VPCMPD + mask
(300 PSI)        │              │            │            │ combine
─────────────────┼──────────────┼────────────┼────────────┼──────────────
JicFitting       │ INT32        │ Bloom      │ ~85B c/s   │ Bloom fast +
(2500 PSI)       │              │ + exact    │            │ exact fallback
─────────────────┼──────────────┼────────────┼────────────┼──────────────
DeepSeaSeal      │ INT32        │ Dual       │ ~40B c/s   │ Compute twice,
(10000 PSI)      │              │ redundant  │            │ XOR masks,
                 │              │ + verify   │            │ assert zero
```

---

## Concrete Implementation Plan

### Phase 1: Intent-Annotated CDCLTrace

Extend `CDCLTrace` to carry intent metadata per constraint:

```rust
pub struct IntentAnnotation {
    pub channel_values: [f64; 9],   // 9-channel profile
    pub tolerance: [f64; 9],         // per-channel tolerance
    pub draft: f64,                   // context depth required
    pub fitting: Fitting,             // HoseClamp/Industrial/JIC/DeepSeaSeal
}

pub struct CDCLTrace {
    pub events: Vec<TraceEvent>,
    pub intent: Vec<Option<IntentAnnotation>>,  // NEW: per-decision intent
}
```

### Phase 2: Intent-Aware Emitter

Extend `build_check_constraints()` to accept intent:

```rust
pub fn build_check_constraints_with_intent(
    intent: &[Option<IntentAnnotation>]
) -> ExecutableBuffer {
    // For each constraint, select code generation strategy:
    // - DeepSeaSeal: dual-redundant + verification
    // - JicFitting: Bloom + exact fallback
    // - IndustrialFitting: INT16 VPCMPD
    // - HoseClamp: INT8 VPCMPD
    //
    // Result: mixed-precision machine code that uses the
    // cheapest sufficient representation for each constraint.
}
```

### Phase 3: Negative Knowledge Fast Path

Add a pre-check that uses Bloom filters to skip safe regions:

```rust
pub fn build_negative_knowledge_check(
    safe_bloom: &[u64; 8],  // 512-bit Bloom filter of known-safe values
    intent: &IntentAnnotation,
) -> ExecutableBuffer {
    // Step 1: Bloom membership test (1 XOR + 1 POPCNT + 1 CMP)
    // Step 2: If Bloom says "probably safe", SKIP the exact check
    // Step 3: If Bloom says "might be unsafe", fall through to exact check
    //
    // For C9 > 0.8 (safety-critical): Skip Bloom entirely
    // For C9 < 0.3 (advisory): Bloom-only, never fall through
}
```

### Phase 4: JIT Intent Adaptation

For long-running systems, the constraint profile changes over time:
- Morning: many constraints, tight tolerances (system warming up)
- Peak: few critical constraints, fast path (performance mode)
- Night: many validation constraints, relaxed timing (audit mode)

The JIT compiler can re-emit machine code based on the CURRENT intent profile:
1. Monitor constraint violation patterns at runtime
2. Adjust tolerances based on observed behavior
3. Re-compile with cheaper code for stable constraints
4. Re-compile with stricter code for unstable constraints

---

## Performance Projections

### Real-World Constraint Mix (Autonomous Vehicle)

| Category | % of Total | Tolerance | Current Code | Intent-Directed |
|----------|-----------|-----------|-------------|-----------------|
| Safety-critical | 2% | < 0.05 | INT32, single check | INT32, dual redundant |
| Navigation | 8% | 0.1–0.3 | INT32, single check | INT32, Bloom+exact |
| Perception | 15% | 0.3–0.5 | INT32, single check | INT16, single check |
| Telemetry | 25% | 0.5–0.8 | INT32, single check | INT8, Bloom-only |
| Logging | 50% | > 0.8 | INT32, single check | INT8, no fallback |

**Current throughput:** 85B constraints/s (everything INT32)
**Intent-directed:** ~340B constraints/s (4x improvement)
**Safety improvement:** 2% of constraints get dual redundancy (currently 0%)

### Why This Works: Amdahl's Law in Reverse

The key insight is that we're not making the slow checks faster — we're making the MAJORITY of checks CHEAPER. 75% of constraints don't need INT32. They're currently paying for precision they don't use.

- 50% of constraints: 4x cheaper (INT8 vs INT32)
- 25% of constraints: 2x cheaper (INT16 vs INT32)
- 15% of constraints: same cost (INT32, same as now)
- 8% of constraints: 0.5x cost (Bloom adds overhead)
- 2% of constraints: 0.25x cost (dual redundancy)
- **Net: ~3.8x throughput improvement** with BETTER safety for the critical 2%

---

## The Proof Path

1. **Benchmark:** Run existing constraint-theory-llvm emitter against mixed constraint set
2. **Classify:** Use LLM encoder to assign 9-channel profiles to each constraint
3. **Re-emit:** Generate mixed-precision machine code based on fitting selector
4. **Verify:** Differential test — same inputs, compare INT8/INT16/INT32 results
5. **Measure:** Throughput (constraints/s), latency (WCET), safety coverage (% dual-redundant)

### What We Can Prove RIGHT NOW on eileen (no GPU needed)

Even without CUDA, we can prove the concept using the x86-64 emitter:
1. Generate 10M constraint triples (value, lower, upper)
2. Classify each by tolerance
3. Emit mixed-precision code (INT8 for 75%, INT32 for 25%)
4. Run through CPU execution
5. Compare results against reference (must be identical)
6. Measure throughput improvement

**This is the proof that makes the science undeniable.**

---

## Connection to Existing Fleet Work

| Component | Current | Intent-Directed |
|-----------|---------|-----------------|
| constraint-theory-llvm | Fixed INT32 AVX-512 | Mixed-precision with fitting selector |
| x86-64 emitter | 3 variants (check/bloom/batch) | Intent-parameterized variants |
| flux-vm (50 opcodes) | Flat bytecode | Annotated bytecode with intent |
| flux-isa (5 variants) | Hardware-specific ISAs | Intent-adaptive ISA selection |
| holonomy-consensus GL(9) | Fleet coordination | Also drives code generation |
| polyformalism-a2a | Communication framework | Also drives compilation strategy |

**The 9-channel model isn't just for communication — it's a COMPILATION DIRECTIVE.**
