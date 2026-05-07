# Intent-Directed Compilation

Using semantic criticality ("stakes") to drive instruction-level precision for constraint checking.

## The Idea

Not all constraints carry equal weight. A water temperature sensor (informational) doesn't need the same verification as a depth sensor (life-critical). We classify constraints by stakes and use narrower integer types for lower-stakes checks, processing 64 per AVX-512 register instead of 16.

## Key Results (Measured, rdtsc, AMD Ryzen AI 9 HX 370)

| Metric | Value |
|--------|-------|
| SoA mixed speedup | **3.17×** (5-run mean) |
| INT8 raw | **4.58×** (exceeds 4.0× theoretical) |
| Differential mismatches | **0 / 100M** |
| Break-even reuses | **8** |
| Steady-state speedup | **12×** at 10K reuses |

## Structure

- `paper/` — IEEE conference paper + research documentation
- `benchmarks/` — Real hardware numbers, cycle-accurate
- `proofs/` — Formal proofs of INT8 soundness and XOR equivalence
- `src/` — C benchmarks, Python validation, ARM NEON fallback

## Proven Theorems

1. **INT8 Soundness:** Cast is identity on [-127, 127] → comparison preserved
2. **XOR Dual-Path:** Sign-bit XOR = adding 2³¹, bijective order isomorphism
3. **dim H⁰ = 9:** Global sections of trivial GL(9) bundle on tree = 9

## Bugs Found by Adversarial Testing

- INT8 overflow wrapping (4.9% mismatch) → fixed with range validation
- Dual-path subtraction overflow → fixed with XOR (6% faster than broken code)

## License

MIT
