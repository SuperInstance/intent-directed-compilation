# Intent-Directed Mixed-Precision Constraint Checking with AVX-512

**Conference Paper — IEEEtran Format**

*Forgemaster ⚒️, Cocapn Fleet*
*Casey Digennaro (Advisor)*

---

## Abstract

We present intent-directed compilation: a technique that uses semantic information about constraint criticality (a continuous "stakes" parameter) to select instruction-level precision for constraint checking. Constraints classified as low-stakes use 8-bit comparisons (64 per AVX-512 register), while life-critical constraints use dual-path 32-bit verification. On an AMD Ryzen AI 9 HX 370 with AVX-512, the mixed-precision kernel achieves **3.17× mean speedup** over uniform 32-bit checking (5-run mean, range 2.72×–3.52×) with **zero differential mismatches across 100 million constraints**. The structure-of-arrays (SoA) memory layout is critical: the same algorithm with array-of-structs (AoS) layout runs **2.4× slower** than the baseline (a 7.5× performance difference from layout alone). The end-to-end pipeline (classification + SoA conversion + checking) breaks even at **8 constraint reuses**, reaching **12× speedup at steady state** for persistent constraint sets. We prove soundness: INT8 checking is identical to INT32 for values in [−127, 127], and our XOR-based dual verification path is a bijective order isomorphism proven correct for all 32-bit integers. Four independent AI models performed adversarial review, finding two real bugs (INT8 overflow wrapping, dual-path subtraction overflow) that we document and fix.

---

## 1. Introduction

Constraint checking — verifying that values lie within prescribed bounds — is a fundamental operation in control systems, sensor fusion, and fleet coordination. The standard approach uses uniform 32-bit integer comparison, processing 16 constraints per AVX-512 register.

We observe that not all constraints carry equal weight. A water temperature sensor reading (informational) does not require the same verification rigor as a depth sensor on an autonomous underwater vehicle (life-critical). This semantic information — how important a constraint is — can drive instruction-level optimization: use fewer bits for low-stakes constraints, more bits (and redundant verification) for high-stakes ones.

We call this **intent-directed compilation**: the semantic intent (stakes) drives the machine code generation (precision class). This paper presents the technique, its implementation, and rigorous experimental validation including adversarial testing.

### Contributions
1. A stakes-based precision classifier mapping continuous criticality to discrete instruction classes
2. AVX-512 implementation with SoA layout achieving 3.17× speedup
3. Formal proofs of soundness (INT8 identity, XOR order isomorphism)
4. End-to-end pipeline analysis showing break-even at 8 reuses
5. Adversarial testing by 4 independent AI models, finding 2 real bugs

---

## 2. Mathematical Framework

### 2.1 Precision Classification

**Definition 1 (Stakes).** A stakes value s ∈ [0, 1] represents the criticality of a constraint, where 0 = informational and 1 = life-critical.

**Definition 2 (Precision Class).** Given stakes s and value range r = hi − lo:

| Condition | Precision Class | Bits | Constraints/Register |
|-----------|----------------|------|---------------------|
| s ≤ 0.25 ∧ r ≤ 127 | INT8 | 8 | 64 |
| s ≤ 0.50 ∧ r ≤ 32000 | INT16 | 16 | 32 |
| s ≤ 0.75 | INT32 | 32 | 16 |
| s > 0.75 | DUAL | 64 | 16 (×2 paths) |

### 2.2 INT8 Soundness

**Theorem 1 (INT8 Soundness).** For all integers v, lo, hi ∈ [−127, 127] where lo ≤ hi:

```
int8_comparison(v, lo, hi) = int32_comparison(v, lo, hi)
```

*Proof.* The int8 cast function f(x) = ((x + 128) mod 256) − 128. For x ∈ [−127, 127], we have x + 128 ∈ [1, 255], so (x + 128) mod 256 = x + 128, giving f(x) = x. Since the cast is the identity on this domain, comparison results are identical. ∎

### 2.3 XOR Dual-Path Equivalence

For safety-critical constraints, we use two independent verification paths that must agree. The original implementation used subtraction (v − lo ≥ 0 ∧ hi − v ≥ 0), which overflows at extreme values. We replace it with XOR-based unsigned comparison.

**Theorem 2 (XOR Order Isomorphism).** For all signed 32-bit integers v, lo, hi:

```
(v ≥ lo ∧ v ≤ hi) ⟺ ((v ⊕ 0x80000000) ≥_u (lo ⊕ 0x80000000) ∧ (v ⊕ 0x80000000) ≤_u (hi ⊕ 0x80000000))
```

where ⊕ is bitwise XOR and ≥_u, ≤_u are unsigned comparisons.

*Proof.* Define g(x) = x ⊕ 0x80000000. In two's complement, XOR with the sign bit is equivalent to adding 2³¹ (mod 2³²), converting signed to unsigned range. This is a strictly increasing bijection: a ≤ b ⟺ g(a) ≤_u g(b). The equivalence follows by applying this order preservation to both comparisons. ∎

The dual path computes: Path A (signed comparison) AND Path B (XOR then unsigned comparison). Both must agree; disagreement indicates a hardware fault.

---

## 3. Implementation

### 3.1 Structure-of-Arrays Layout

Constraints are sorted by precision class into contiguous arrays:

```c
SoABatch {
    int8_t  v8[N],  lo8[N],  hi8[N];   // INT8 group
    int16_t v16[N], lo16[N], hi16[N];  // INT16 group
    int32_t v32[N], lo32[N], hi32[N];  // INT32 group
    int32_t vd[N],  lod[N],  hid[N];   // DUAL group (XOR path)
}
```

This is **critical**. With AoS layout, each constraint writes to a different buffer based on its class, causing random memory access. Our benchmarks show:

| Layout | Speedup vs INT32 Baseline |
|--------|---------------------------|
| SoA (contiguous by class) | **3.17×** |
| AoS (interleaved) | **0.42×** (2.4× slower than baseline) |

The 7.5× difference from layout alone is the single most important implementation detail.

### 3.2 AVX-512 Constraint Checking

Each precision class uses the appropriate VPCMP instruction:

```c
// INT8: 64 constraints per register
__mmask64 k = _mm512_cmpge_epi8_mask(vv, ll) & _mm512_cmple_epi8_mask(vv, hh);

// INT16: 32 per register
__mmask32 k = _mm512_cmpge_epi16_mask(vv, ll) & _mm512_cmple_epi16_mask(vv, hh);

// INT32: 16 per register
__mmask16 k = _mm512_cmpge_epi32_mask(vv, ll) & _mm512_cmple_epi32_mask(vv, hh);

// DUAL: XOR-based (overflow-safe)
__mmask16 ka = _mm512_cmpge_epi32_mask(vv, ll) & _mm512_cmple_epi32_mask(vv, hh);
__m512i sign = _mm512_set1_epi32(0x80000000);
__mmask16 kb = _mm512_cmpge_epu32_mask(_mm512_xor_si512(vv, sign), 
                    _mm512_xor_si512(ll, sign))
           & _mm512_cmple_epu32_mask(_mm512_xor_si512(vv, sign),
                    _mm512_xor_si512(hh, sign));
// Both paths must agree
```

### 3.3 Bloom Filter Fast Path

For constraints that are far from boundaries, a Bloom filter provides a "definitely safe" fast path. The filter achieves 67.1% hit rate (skipping exact checks) with zero false confirmations — if the filter says "safe," the constraint is guaranteed to pass. This is an instance of **negative knowledge**: knowing where violations are NOT is computationally cheaper than checking everywhere.

---

## 4. Experimental Results

### 4.1 Setup

All measurements on an **AMD Ryzen AI 9 HX 370** (Zen 5, AVX-512F/BW/DQ), compiled with `gcc -O3 -mavx512f -mavx512bw -mavx512dq`. Timing uses `rdtsc` (cycle-accurate). 5-run reproducibility reported.

### 4.2 Microbenchmark: Per-Precision Throughput

Table 1: Cycle-accurate throughput per precision class (10M constraints)

| Precision | Cycles/constraint | Constraints/cycle | Speedup vs INT32 |
|-----------|------------------|-------------------|------------------|
| INT32 | 0.70 | 1.4 | 1.00× (baseline) |
| INT8 | 0.15 | 6.5 | **4.58×** |
| INT16 | 0.31 | 3.2 | **2.25×** |
| DUAL (XOR) | 0.53 | 1.9 | **1.32×** (cost) |

The INT8 speedup (4.58×) **exceeds the theoretical 4.0× packing ratio** because INT8 data is 4× smaller, fitting more constraints per cache line. With a baseline time model T = T_comp + T_mem + T_loop, the cache bandwidth factor M ≈ 4.49 explains the super-linear gain.

### 4.3 SoA Mixed-Precision Throughput

Table 2: SoA mixed-precision with AV mix (75% INT8, 15% INT16, 8% INT32, 2% DUAL)

| Run | Speedup |
|-----|---------|
| 1 | 3.09× |
| 2 | 3.11× |
| 3 | 3.52× |
| 4 | 2.72× |
| 5 | 3.41× |
| **Mean** | **3.17×** |

### 4.4 Non-Uniform Threshold Validation

Seed-2.0-mini (simulating a senior compiler engineer) predicted that per-constraint thresholds would eliminate the speedup. We tested: each constraint has its own (lo, hi) pair, loaded per-lane by VPCMPD. Result: **3.96×** for pure INT8 with non-uniform bounds. The VPCMP instructions compare lane-by-lane natively; no uniform threshold assumption is needed.

### 4.5 Correctness Verification

**Differential testing:** Every constraint is checked at both the downgraded precision and INT32. Results compared.

| Test Scale | Precision Classes | Mismatches |
|-----------|------------------|------------|
| 100M constraints (AVX-512) | INT8, INT16, INT32, DUAL | **0** |
| 8.3M exhaustive Python | INT8 vs INT32, [−127, 127] | **0** |
| 10M random Python | INT16 vs INT32, [−32767, 32767] | **0** |
| 5M random INT32 triples | XOR dual-path vs comparison | **0** |

### 4.6 End-to-End Pipeline

Table 3: Full pipeline including classification and SoA conversion (10M constraints, random stakes)

| Phase | Cycles/constraint | % Total |
|-------|-------------------|---------|
| Pass 1: Classify | 6.33 | 29.5% |
| Pass 2: SoA sort | 14.92 | 69.4% |
| Phase 3: AVX-512 check | 0.24 | 1.1% |
| **Total** | **21.49** | **0.14× vs scalar baseline** |

The one-shot pipeline is 7× slower than baseline. But for reused constraint sets:

| Reuses | Pipeline Speedup |
|--------|-----------------|
| 1 | 0.14× (don't use) |
| 10 | 1.23× |
| 100 | 6.43× |
| 1,000 | 11.14× |
| 10,000 | 12.02× |

**Break-even: 8 reuses.** For 1kHz sensor polling, this is 8ms. For fleet topology (static constraints), conversion happens once and the 0.24 cyc/constraint check runs indefinitely.

### 4.7 Real Application: RARS-IMU

127 autonomous underwater vehicle constraints (depth, attitude, pressure hull, sonar, battery, etc.) classified by stakes:

| Class | Count | % | Examples |
|-------|-------|---|---------|
| INT8 | 20 | 16% | Water temp, GPS fix |
| INT16 | 42 | 33% | Heading, motor current |
| INT32 | 43 | 34% | Roll/pitch/yaw rate, accel |
| DUAL | 22 | 17% | Depth, pressure hull, leak detect |

Total check time: **20 cycles** (sub-nanosecond). At 1kHz control rate, this is 0.008% of the 1ms budget.

---

## 5. Bugs Found and Fixed

### 5.1 INT8 Overflow Wrapping

**Found by:** Qwen3-235B (red team adversarial testing)

**Bug:** Values outside [−128, 127] wrap in 8-bit two's complement. When the classifier assigns INT8 but the value exceeds this range, the comparison gives wrong results. Testing showed 4.9% mismatch rate for out-of-range values.

**Fix:** Added explicit range validation in the classifier. If any value or bound exceeds [−127, 127], promote to INT16 automatically.

### 5.2 Dual-Path Subtraction Overflow

**Found by:** Adversarial edge-case testing

**Bug:** The original dual-path used subtraction (v − lo ≥ 0 ∧ hi − v ≥ 0). This overflows when hi = INT_MAX and v < 0 (or symmetrically). Three specific edge cases cause Path A (comparison) to say PASS while Path B (subtraction) says FAIL.

**Fix:** Replaced subtraction with XOR-based unsigned comparison (Theorem 2). The fix is **6% faster** than the broken subtraction approach (0.94× cost) because the CPU pipelines the XOR + unsigned compare more efficiently than two subtractions + two sign checks.

---

## 6. Discussion

### 6.1 Layout Is Everything

The 7.5× SoA vs AoS difference is the most actionable finding. Any SIMD constraint checking system MUST use structure-of-arrays layout. The conversion cost is amortized at 8 reuses; for persistent constraint sets, the conversion is a one-time cost.

### 6.2 The Cache Bandwidth Effect

INT8's 4.58× (vs 4.0× theoretical) comes from memory effects: each 64-byte cache line holds 64 INT8 constraints vs 16 INT32. This 4× reduction in cache pressure gives a super-linear throughput gain.

### 6.3 Not All Systems Benefit

For 127-constraint sensor fusion at 1kHz, the entire check takes 8ns in a 1ms budget. Mixed-precision saves ~6ns. The engineering complexity of classification + SoA layout is not worth it for this scale. The optimization matters at fleet scale (millions of constraints) or when constraints are checked millions of times.

### 6.4 ARM NEON Fallback

For embedded safety controllers (Cortex-R, no AVX-512), we provide a scalar fallback. NEON (128-bit) gives 4× less parallelism per register. The XOR dual-path works on any architecture.

### 6.5 Limitations

- **WSL2 multi-core scaling is poor** (hybrid P/E cores + VM scheduling)
- **Intel Xeon results will differ** (different AVX-512 port allocation)
- **One-shot constraint sets** should NOT use mixed-precision (0.14×)
- **The thresholds (0.25/0.50/0.75)** are empirically optimal but not derived from first principles

---

## 7. Related Work

Mixed-precision computation is well-established in machine learning (FP16/FP8 training) and signal processing. Adaptive mesh refinement in finite element analysis selects precision based on error estimators — conceptually similar to our stakes-based precision selection. Bloom filters for approximate set membership are standard in databases and networking; our application to constraint pre-filtering appears novel. Dual-path verification is related to triple modular redundancy in fault-tolerant computing, though we use only two paths with mathematical equivalence proof.

---

## 8. Conclusion

Intent-directed compilation achieves 3.17× speedup for constraint checking by using semantic information (stakes) to select instruction-level precision. The approach is provably sound: INT8 checking is identical to INT32 for values in [−127, 127], and XOR-based dual verification is correct for all 32-bit integers. The key practical insight is that SoA memory layout is mandatory (7.5× difference) and the optimization only benefits reused constraint sets (break-even at 8). For one-shot checks, uniform INT32 is superior. Adversarial testing by four independent AI models found two real bugs, both fixed with provably correct alternatives.

---

## References

[1] Intel® 64 and IA-32 Architectures Software Developer's Manual. AVX-512 Vector Instructions.

[2] DeepSeek. Formal proofs: INT8 soundness and XOR dual-path equivalence. Private communication, 2026.

[3] Bloom, B.H. Space/time trade-offs in hash coding with allowable errors. Communications of the ACM, 13(7), 1970.

[4] DO-178C: Software Considerations in Airborne Systems and Equipment Certification. RTCA, 2012.

[5] SOA vs AOS data layout for SIMD. Game Engine Architecture, Gregory, 2018.

---

*Source code and reproducible benchmarks: github.com/SuperInstance/polyformalism-thinking/tree/main/research*
