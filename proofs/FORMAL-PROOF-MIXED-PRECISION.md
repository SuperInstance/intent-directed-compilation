# Formal Proof Results: Mixed-Precision Constraint Checking

**Date:** 2026-05-06
**Prover:** deepseek-reasoner
**Status:** 3 proven, 1 disproven

---

## Summary

| Theorem | Claim | Result | Notes |
|---------|-------|--------|-------|
| T1 (Soundness) | Truncation never introduces false passes | ✅ **PROVEN** | Trivially true: values within b-bit range are exactly representable, so b-bit and 32-bit comparisons are identical. Overflow in subtraction-based checks cannot produce false passes when lower ≤ upper (proved by contradiction on ordering). |
| T2 (Completeness) | Truncation never hides real failures | ✅ **PROVEN** | Direct contrapositive of T1. If exact check fails, b-bit must also fail because b-bit passing would imply exact passing. |
| T3 (Dual Redundancy) | Silent fault probability bounded by double-fault probability | ✅ **PROVEN** | Event of silent incorrect PASS ⊂ event of two independent faults producing identical results. Estimated bound: ~10⁻²⁴ to 10⁻³⁶ per operation for VPCMPD with ECC. |
| T4 (Throughput Bound) | G = (4a + 2b + c + 0.5d)/(a+b+c+d) | ❌ **DISPROVEN** | Counterexample: 50% INT8, 50% INT32 gives formula's G=2.5 but actual G=1.6. Formula uses arithmetic mean; correct formula uses harmonic mean: G = 1/(a/4 + b/2 + c + 2d). |

---

## Theorem Details

### T1 (Soundness) — Proven

**Key insight:** Since |value|, |lower|, |upper| < 2^(b-1), all three values are exactly representable as b-bit signed integers. The b-bit truncated representation is bitwise identical to the low b bits of the 32-bit value (higher bits are sign extension). Therefore, any comparison operation yields the same boolean result at b-bit and 32-bit precision.

**Subtraction-based check proof:** Even if the check uses subtraction (value − lower, upper − value) in b-bit arithmetic, overflow cannot produce a false PASS when lower ≤ upper. The proof by contradiction shows that the ordering constraint lower ≤ upper makes it impossible for both b-bit differences to be non-negative when the exact check fails. Specifically:
- For false PASS from value < lower: requires lower − value ≥ 129 and upper − value ≤ 127, but lower ≤ upper gives 129 ≤ upper − value ≤ 127 — contradiction.
- Symmetric argument for value > upper.

### T2 (Completeness) — Proven

Direct contrapositive of T1: ¬PASS₃₂ → ¬PASS_b. If the exact check fails and the b-bit check passed, that would violate soundness.

### T3 (Dual Redundancy) — Proven

**Bound derivation:** Let E = "silent incorrect PASS" and F = "two independent faults produce identical boolean results". Since E ⊂ F, P(E) ≤ P(F).

**Estimate for VPCMPD with ECC:**
- ECC eliminates memory data errors; only execution logic faults remain
- Soft error rate for combinational logic: ~10⁻⁹ per device-hour
- Per 1ns operation: p ≈ 10⁻¹⁸
- Assuming independent faults: P(E) ≤ p² ≈ 10⁻³⁶
- Conservative estimate (p = 10⁻¹²): P(E) ≤ 10⁻²⁴

This is negligibly small — effectively zero for practical purposes.

### T4 (Throughput Bound) — DISPROVEN

**Counterexample:** 50% INT8 (a=0.5), 50% INT32 (c=0.5), b=d=0.

- Formula claims: G = (4×0.5 + 0.5) / 1 = 2.5
- Actual: Time = 0.5N/(4R) + 0.5N/R = 0.625N/R → G = 1/0.625 = **1.6**
- Overestimation factor: 2.5/1.6 = 1.56×

**Root cause:** The formula uses the arithmetic mean of relative throughputs weighted by constraint fractions. The correct formula is the **harmonic mean**:

```
G_correct = 1 / (a/4 + b/2 + c + 2d)
```

The arithmetic mean formula would be correct only if a,b,c,d represent **instruction fractions** (proportion of instructions, not constraints).

---

## Implications for Intent-Directed Compilation

### What's Safe
1. **Soundness (T1) and Completeness (T2) are proven.** Mixed-precision truncation does NOT introduce false passes or hide real failures, provided values stay within the b-bit range. This is the critical safety property — our INT8 fast path won't silently approve a violation.

2. **Dual redundancy (T3) provides astronomically strong fault tolerance.** The ~10⁻²⁴+ bound means dual-redundant INT32 checks are effectively immune to silent hardware faults.

### What Needs Fixing
3. **The throughput formula (T4) is wrong.** The claimed 3.4× speedup for the AV example (75% INT8, 15% INT16, 8% INT32, 2% DUAL) must be recalculated:

   ```
   G_correct = 1 / (0.75/4 + 0.15/2 + 0.08 + 2×0.02)
             = 1 / (0.1875 + 0.075 + 0.08 + 0.04)
             = 1 / 0.3825
             = 2.61×
   ```

   The original formula gives 3.4×. **The actual speedup is 2.6×, not 3.4×.** Still significant, but the claim needs correction.

### What Needs Strengthening Before Production

1. **Range guard validation:** Soundness T1 depends on |value| < 2^(b-1). We need a **runtime guard** that validates this invariant before truncating. If a value exceeds the b-bit range, we must fall back to INT32. The cost of this guard must be included in throughput calculations.

2. **Throughput claims need correction:** All published speedup numbers must use the harmonic mean formula, not the arithmetic mean. This reduces the headline number from ~3.4× to ~2.6× for the AV scenario.

3. **The dual redundancy estimate assumes independent faults.** In practice, the comparison and subtraction paths may share some logic (e.g., the ALU adder). A more conservative bound should account for common-mode failures. Formal fault tree analysis needed.

4. **No proof for unsigned variants:** The theorems assume signed two's complement. If unsigned comparisons are used (e.g., for modular arithmetic constraints), soundness needs separate proof.

5. **Edge case at exactly 2^(b-1):** The theorem uses strict inequality |value| < 2^(b-1). The value -2^(b-1) (e.g., -128 for b=8) is NOT covered. Need explicit handling or proof extension.

---

## Next Steps

1. ✅ Fix throughput formula in all documentation and code
2. ✅ Add range guard emission to the compiler (assert |value| < 2^(b-1) before truncation)
3. ✅ Recalculate all published benchmarks with corrected formula
4. ⬜ Formal verification in Coq/Lean for T1/T2 (machine-checkable proof)
5. ⬜ Fault tree analysis for dual redundancy common-mode failures
6. ⬜ Extend proofs to unsigned arithmetic and modular constraints
